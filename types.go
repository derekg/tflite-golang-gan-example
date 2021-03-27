package main

/*
#include "tensorflow/lite/c/c_api.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

const float32Size = 4

func TfLiteTensorCopyFromBuffer(p0 *C.TfLiteTensor, p1 unsafe.Pointer, p2 int) C.TfLiteStatus {
	return C.TfLiteTensorCopyFromBuffer(p0, p1, C.ulong(p2))
}

func TfLiteTensorCopyToBuffer(p0 *C.TfLiteTensor, p1 unsafe.Pointer, p2 int) C.TfLiteStatus {
	return C.TfLiteTensorCopyToBuffer(p0, p1, C.ulong(p2))
}
