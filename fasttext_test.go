package fasttextgo

import (
	"fmt"
	"testing"
)

func TestSimple(t *testing.T) {
	LoadModel("fasttext_search_category.clf.CLF.bin")
	fmt.Println(Predict("sushi in palo alto"))
	fmt.Println(Predict("thriller movies"))
}
