package main

/*

#cgo CFLAGS: -I ./include

//points to the right platform version of tflite libs
#cgo arm LDFLAGS: -L arm
#cgo darwin LDFLAGS: -L macosx
#cgo x86_64 LDFLAGS: -L x86_64

#cgo LDFLAGS: -ltensorflowlite_c

//Raspberry Pi needs to include libatomic when linking w/ tflite
#cgo arm LDFLAGS: -latomic

#include "tensorflow/lite/c/c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"math/rand"
	"net/http"
	"sync"
	"time"
	"unsafe"
)

const seedSize = 512
const imgSize = 128
const imgChannels = 3

type TFGan struct {
	modelName    *C.char
	model        *C.TfLiteModel
	options      *C.TfLiteInterpreterOptions
	runner       *C.TfLiteInterpreter
	input        *C.TfLiteTensor
	inputBuffer  [seedSize]float32
	output       *C.TfLiteTensor
	outputBuffer [imgSize * imgSize * imgChannels]float32
	mutex        sync.Mutex
}

func makeTFGan(modelName string) *TFGan {
	version := C.TfLiteVersion()
	fmt.Printf("Tensorflow Version: %v\n", C.GoString(version))
	name := C.CString(modelName)
	model := C.TfLiteModelCreateFromFile(name)
	if model == nil {
		fmt.Printf("failed to create model from - %v\n", C.GoString(name))
		return nil
	}
	options := C.TfLiteInterpreterOptionsCreate()
	if options == nil {
		fmt.Printf("failed to create options for %v\n", modelName)
		return nil
	}
	runner := C.TfLiteInterpreterCreate(model, options)
	if runner == nil {
		fmt.Printf("failed to create interperter for %v\n", modelName)
		return nil
	}
	C.TfLiteInterpreterAllocateTensors(runner)
	input := C.TfLiteInterpreterGetInputTensor(runner, 0)
	if input == nil {
		fmt.Printf("input tensor is empty\n")
		return nil
	}
	output := C.TfLiteInterpreterGetOutputTensor(runner, 0)
	if output == nil {
		fmt.Printf("putput tensor is empty\n")
		return nil
	}

	return &TFGan{
		modelName:    name,
		model:        model,
		options:      options,
		runner:       runner,
		input:        input,
		output:       output,
		outputBuffer: [imgSize * imgSize * imgChannels]float32{},
		inputBuffer:  [seedSize]float32{},
	}
}

func (gan *TFGan) free() {
	C.TfLiteInterpreterDelete(gan.runner)
	C.TfLiteModelDelete(gan.model)
	C.free(unsafe.Pointer(gan.modelName))
}

func (gan *TFGan) generate() image.Image {
	gan.mutex.Lock()
	defer gan.mutex.Unlock()
	for i := 0; i < len(gan.inputBuffer); i++ {
		gan.inputBuffer[i] = rand.Float32()
	}
	tfseed := unsafe.Pointer(&gan.inputBuffer)
	m := TfLiteTensorCopyFromBuffer(gan.input, tfseed, len(gan.inputBuffer)*float32Size)
	if m != C.kTfLiteOk {
		size := C.TfLiteTensorByteSize(gan.input)
		fmt.Printf("failed to copy buffer - size : %v -  %v\n", size, m)
		return nil
	}
	if C.TfLiteInterpreterInvoke(gan.runner) != C.kTfLiteOk {
		fmt.Printf("failed to run\n")
		return nil
	}

	m = TfLiteTensorCopyToBuffer(gan.output, unsafe.Pointer(&gan.outputBuffer), len(gan.outputBuffer)*float32Size)
	if m != C.kTfLiteOk {
		size := C.TfLiteTensorByteSize(gan.output)
		fmt.Printf("failed to copy ouput buffer - size : %v -  %v\n", size, m)
	}

	img := image.NewRGBA(image.Rect(0, 0, imgSize, imgSize))
	flip := rand.Int31n(2)
	for y := 0; y < imgSize; y++ {
		for x := 0; x < imgSize; x++ {
			offset := (y * imgSize * imgChannels) + (x * imgChannels)
			color := img.RGBAAt(x, y)
			color.R = uint8(gan.outputBuffer[offset] * 255.0)
			color.G = uint8(gan.outputBuffer[offset+1] * 255.0)
			color.B = uint8(gan.outputBuffer[offset+2] * 255.0)
			color.A = 255
			//hack to flip which direction the face is facing
			if flip > 0 {
				img.SetRGBA(x, y, color)
			} else {
				img.SetRGBA(127-x, y, color)
			}
		}
	}
	return img
}

func main() {
	var modelName string
	var port string
	flag.StringVar(&modelName, "m", "", "tflite model name")
	flag.StringVar(&port, "p", "8080", "port to listen on ")
	flag.Parse()
	if modelName == "" {
		flag.Usage()
		return
	}
	rand.Seed(time.Now().Unix())
	gan := makeTFGan(modelName)
	if gan == nil {
		fmt.Printf("failed \n")
		return
	}

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-cache, private, max-age=0")
		w.Header().Set("Expires", time.Unix(0, 0).Format(http.TimeFormat))
		img := gan.generate()
		png.Encode(w, img)
	})
	fmt.Printf("Listenting on %v\n", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		fmt.Printf("failed to listen on :%v for http requests %v\n", port, err)
	}
	gan.free()
}
