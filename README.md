# tflite-golang-gan-example

This a simple example of using Golang w/ TFLite to be able to easily run simple models.  All the bits and pieces are included in the repo

For Linux/MacOs amd64

```go build example.go types.go```

For Raspberry Pi 

```go build example.go types_arm.go```

This is build and running is  

```LD_LIBRARY_PATH=and64/macos/arm ./example -m model/model.tflite```

This will setup a listener on localhost:8080 for you to view the results. 

Check https://derekg.github.io/tflite.html for more details. 
