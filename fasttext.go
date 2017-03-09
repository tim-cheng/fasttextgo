package fasttextgo

// #cgo LDFLAGS: -L${SRCDIR} -lfasttext -lstdc++
// #include <stdlib.h>
// void load_model(char *path);
// int predict(char *query, float *prob, char *buf, int buf_sz);
import "C"
import (
	"errors"
	"unsafe"
)

// LoadModel - load FastText model
func LoadModel(path string) {
	C.load_model(C.CString("fasttext_search_category.clf.CLF.bin"))
}

// Predict - predict
func Predict(sentence string) (prob float32, label string, err error) {

	var cprob C.float
	var buf *C.char
	buf = (*C.char)(C.calloc(64, 1))

	ret := C.predict(C.CString(sentence), &cprob, buf, 64)

	if ret != 0 {
		err = errors.New("error in prediction")
	} else {
		label = C.GoString(buf)
		prob = float32(cprob)
	}
	C.free(unsafe.Pointer(buf))

	return prob, label, err
}
