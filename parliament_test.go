package main

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestBase(t *testing.T) {
	// This one will panic if there's a problem.
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("Recovered in f", r)
		}
	}()
	x := New([]int{3, 4})
	x.Project()
	x.Eval([]int{1, 1, 1})
	x.Score([]int{1, 0, -1}, []int{0, 1, 0, -1})
	x.BProp([][]int{[]int{1, 0, -1}}, [][]int{[]int{0, 1, 0, -1}}, 5.0)
	t.Log(promdump(1))
}

func TestSimple(t *testing.T) {
	x := New([]int{3, 4})
	i := 0
	for i = 0; i < 10000; i++ {
		inp := [][]int{[]int{1, 1, 1}, []int{-1, 1, -1}}
		out := [][]int{[]int{1, 1, 1, 1}, []int{-1, 1, 1, -1}}
		x.BProp(inp, out, 5.0)
	}
	t.Log(promdump(1))

}
func TestComplex(t *testing.T) {
	x := New([]int{3, 4})
	// a complicated test
	for i := 0; i < 10000; i++ {
		shape := 10
		inp := make([][]int, shape, shape)
		out := make([][]int, shape, shape)
		for j := 0; j < shape; j++ {
			a := (rand.Int() % 3) - 1
			b := (rand.Int() % 3) - 1
			c := (rand.Int() % 3) - 1
			inp[j] = []int{a, b, c}
			out[j] = []int{-a, -1, -1, b}

		}
		x.BProp(inp, out, 5.0)
	}
	t.Log(promdump(1))
}

func TestMultilayer(t *testing.T) {
	x := New([]int{3, 6, 5})
	// A simple test
	for i := 0; i < 10000; i++ {
		inp := [][]int{[]int{1, 1, 1}}
		out := [][]int{[]int{1, 1, 1, 1, -1}}
		x.BProp(inp, out, 5.0)
	}
	t.Log(promdump(1))

}

func bitrep(i, n int) []int {
	out := make([]int, n, n)
	for loc := range out {
		out[loc] = (i%2)*2 - 1
		i = (i - (i % 2)) / 2
	}
	return out
}

func TestXor(t *testing.T) {
	base := 1
	x := New([]int{2 * base, 2 * (base + 1), 2 * (base + 1), base})
	x.Randomize()
	max := 1
	for i := 0; i < base; i++ {
		max = max * 2
	}
	// a complicated test
	for i := 0; i < 1000; i++ {
		shape := 4
		inp := make([][]int, shape, shape)
		out := make([][]int, shape, shape)
		for j := 0; j < shape; j++ {
			a := rand.Int() % max
			b := rand.Int() % max

			ab := bitrep(a, base)
			bb := bitrep(b, base)
			inp[j] = append(ab, bb...)
			temp := make([]int, base, base)
			for kk := range temp {
				temp[kk] = (ab[kk] + bb[kk]) % 2
			}
			out[j] = temp

		}
		x.BProp(inp, out, 5.0*1e7/(float64(i)-1e7))
	}
	t.Log(promdump(1))

}

func TestAdder(t *testing.T) {
	base := 1
	x := New([]int{2 * base, (base + 1) * (base + 1), base + 1})
	x.Randomize()
	max := 1
	for i := 0; i < base; i++ {
		max = max * 2
	}
	// a complicated test
	for i := 0; i < 10000; i++ {
		shape := int(max/2 + 1)
		inp := make([][]int, shape, shape)
		out := make([][]int, shape, shape)
		for j := 0; j < shape; j++ {
			a := rand.Int() % max
			b := rand.Int() % max

			ab := bitrep(a, base)
			bb := bitrep(b, base)
			inp[j] = append(ab, bb...)
			out[j] = bitrep(a+b, base+1)

		}
		x.BProp(inp, out, 5.0*1e7/(float64(i)-1e7))
	}
	t.Log(promdump(1))
}
