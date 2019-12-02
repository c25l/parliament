package main

import (
	"bytes"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"time"

	"github.com/petar/GoMNIST"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/expfmt"
)

var (
	// track vb vs actual magnitudes, maybe a histogram?
	pReg = prometheus.NewRegistry()
	met  = prometheus.NewSummaryVec(prometheus.SummaryOpts{
		Name:       "test",
		Help:       "testing information",
		Objectives: map[float64]float64{},
	}, []string{"_"})
)

func init() {
	pReg.MustRegister(met)
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	go func() {
		for _ = range c {
			log.Println("pid for kill: ", os.Getpid())
			log.Println(promdump(0))
		}
	}()

	//rand.Seed(time.Now().UnixNano())
}

// Layer is a container for minimal voting units of the network..
type Layer struct {
	Gates [][]int
}

// Net is a container for layers.
type Net struct {
	Layers []Layer
}

func project(x int) int {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}
func fProject(x float64) int {
	if x > 1e-10 {
		return 1
	} else if x < -1e-10 {
		return -1
	}
	return 0
}

func hProject(x int) int {
	if x >= 0 {
		return 1
	}
	return -1

}

// Build a network with the layer counts listed. It will work like []int{input, hidden, hidden..., output}
func New(layers []int) *Net {
	out := Net{make([]Layer, len(layers)-1, len(layers)-1)}
	for ii, ins := range layers {
		if ii >= len(layers)-1 {
			continue
		}
		outs := layers[ii+1]
		out.Layers[ii].Gates = make([][]int, outs, outs)
		for jj := 0; jj < outs; jj++ {
			out.Layers[ii].Gates[jj] = make([]int, ins+1, ins+1)
		}
	}
	return &out
}

// Randomize the state of the net to something nonzero (hopefully)
func (n *Net) Randomize() {
	for ii := range (*n).Layers {
		for jj := range (*n).Layers[ii].Gates {
			for kk := range (*n).Layers[ii].Gates[jj] {
				x := (rand.Int() % 3) - 1
				(*n).Layers[ii].Gates[jj][kk] = x
			}
		}
	}
}

// Project the state of the net to something nonzero (hopefully)
func (n *Net) Project() {
	for ii := range (*n).Layers {
		for jj := range (*n).Layers[ii].Gates {
			for kk := range (*n).Layers[ii].Gates[jj] {
				temp := (*n).Layers[ii].Gates[jj][kk]
				(*n).Layers[ii].Gates[jj][kk] = project(temp)
			}
		}
	}
}

// Eval finds the output for a given input
func (l Layer) Eval(inputs []int) []int {
	if len(inputs) != len(l.Gates[0])-1 {
		log.Println(len(inputs), len(l.Gates[0]))
		panic("irreconcilable inputs and layer layout")
	}
	output := make([]int, len(l.Gates), len(l.Gates))
	for ii, g := range l.Gates {
		if len(g)-1 != len(inputs) {
			panic("irreconcilable gate weights and inputs")
		}
		temp := int(0)
		for ii := range inputs {
			temp += inputs[ii] * g[ii]
		}
		temp += g[len(g)-1]
		output[ii] = project(temp)
	}
	return output
}

// Eval finds the output for a given input
func (n *Net) Eval(inputs []int) [][]int {
	if len(inputs) != len((*n).Layers[0].Gates[0])-1 {
		log.Println(len(inputs), len((*n).Layers[0].Gates[0]))
		panic("bad input dimension")
	}
	carrier := make([][]int, len((*n).Layers), len((*n).Layers))
	carrier[0] = (*n).Layers[0].Eval(inputs)
	for ii := 1; ii < len((*n).Layers); ii++ {
		carrier[ii] = (*n).Layers[ii].Eval(carrier[ii-1])
	}
	return carrier
}

// Score compares the actual output with the intended output.
func (n *Net) Score(input, output []int) []int {
	proto := (*n).Eval(input)
	result := proto[len(proto)-1]
	if len(result) != len(output) {
		panic("cannot get result from specified net")
	}
	ret := make([]int, len(output), len(output))
	good := 0
	for ii := range result {
		ret[ii] = project(output[ii] - result[ii])
		if ret[ii] == 0 {
			good++
		}

	}
	met.WithLabelValues("score_easy").Observe(float64(good) / float64(len(result)))
	if good == len(result) {
		met.WithLabelValues("score_hard").Observe(1.0)
	} else {
		met.WithLabelValues("score_hard").Observe(0.0)
	}
	return ret
}

// FlipNodes sets nodes to random values with exponential frequency.
func (n *Net) FlipNodes(index int, input [][]float64, rate, bkgRate float64) float64 {
	sum := 0.0
	for ia, va := range input {
		if ia >= len((*n).Layers[index].Gates) {
			continue
		}
		for ib, vb := range va {
			if ia >= len((*n).Layers[index].Gates[ia]) {
				continue
			}
			sum += math.Abs(vb)
			if rand.ExpFloat64() > (rate+bkgRate)/(math.Abs(vb)+bkgRate) {
				if math.Abs(vb) < bkgRate {
					met.WithLabelValues("bkgEntry").Observe(1.)
				} else {
					met.WithLabelValues("bkgEntry").Observe(0.)
				}
				if vb == 0 {
					vb = float64((rand.Int()%2)*2 - 1)
					met.WithLabelValues("rand").Observe(1.)
				} else {
					met.WithLabelValues("rand").Observe(0.)
				}
				temp := (*n).Layers[index].Gates[ia][ib]
				met.WithLabelValues("bpr").Observe(vb)
				if project(temp) == fProject(vb) {
					met.WithLabelValues("match").Observe(float64(project(temp)))
				}
				if temp > 0 {
					met.WithLabelValues("pos").Observe(float64(temp))
				} else {
					met.WithLabelValues("neg").Observe(float64(-temp))
				}
				(*n).Layers[index].Gates[ia][ib] = fProject(float64(temp) + vb)
			}
		}
	}
	return sum
}

// BProp implements Backprop on a layer-by-layer level
func (l Layer) BProp(diff []float64) [][]float64 {
	diffs := make([][]float64, len(diff), len(diff))
	for xx := range diffs {
		diffs[xx] = make([]float64, len(l.Gates[0]), len(l.Gates[0]))
	}
	for jj := 0; jj < len(l.Gates); jj++ {
		gate := l.Gates[jj]
		for kk := range gate {
			//restrict to directions you can actually move.
			if diff[jj] > 0 {
				if gate[kk] <= 0 {
					diffs[jj][kk] += diff[jj]
				}
			} else {
				if gate[kk] >= 0 {
					diffs[jj][kk] += diff[jj]
				}
			}
		}
	}
	return diffs
}

// BProp implements the backprop algorithm in this domain.
func (n *Net) BProp(inp, expected [][]int, rate, bkgRate float64) float64 {
	if len(inp) != len(expected) {
		panic("cannot backprop without equal dimensions")
	}
	// Computed desired gradients
	temp := make([][]int, len(inp), len(inp))
	for xx := range inp {
		temp[xx] = n.Score(inp[xx], expected[xx])
	}
	diff := make([]float64, len(temp[0]), len(temp[0]))
	for yy := range temp {
		for xx := range temp[yy] {
			diff[xx] += float64(temp[yy][xx])
		}
	}
	out := 0.
	for xx := len((*n).Layers) - 1; xx >= 0; xx-- {
		bprop := (*n).Layers[xx].BProp(diff)
		out += n.FlipNodes(xx, bprop, rate, bkgRate)
		diff = make([]float64, len(bprop[0]), len(bprop[0]))
		for ii := range diff {
			for jj := range bprop {
				diff[ii] += bprop[jj][ii]
			}
		}
		for ii := range diff {
			diff[ii] /= float64(len(bprop))
		}

		met.WithLabelValues(fmt.Sprintf("tog_%d", xx)).Observe(out)
	}
	for ii, xx := range inp {
		score := n.Score(xx, expected[ii])
		good := 0
		for _, jj := range score {
			if jj == 0 {
				good++
			}
		}
		met.WithLabelValues("eval_easy").Observe(float64(good) / float64(len(xx)))
		if good == len(xx) {
			met.WithLabelValues("eval_hard").Observe(1.0)
		} else {
			met.WithLabelValues("eval_hard").Observe(0.0)
		}

	}
	return out
}

func onehot(n, max int) []int {
	out := make([]int, max, max)
	for xx := range out {
		if n == xx {
			out[xx] = 1
		} else {
			out[xx] = -1
		}

	}
	return out
}

func smash(inp []byte) []int {
	out := make([]int, len(inp), len(inp))
	for xx, yy := range inp {
		if yy > 10 {
			out[xx] = 1
		} else {
			out[xx] = -1
		}
	}
	return out
}

func promdump(kill int) string {
	mfs, _ := pReg.Gather()
	writer := bytes.NewBuffer([]byte{})
	enc := expfmt.NewEncoder(writer, "text/plain; version=0.0.4")
	for _, mf := range mfs {
		enc.Encode(mf)
	}
	if kill > 0 {
		met.Reset()
	}
	return writer.String()
}

func argmax(i []int) (int, int) {
	amx := -1
	max := -2
	count := 0
	for ii, xx := range i {
		if xx > max {
			max = xx
			amx = ii
			count = 1
		}
		if xx == max {
			count++
		}
	}
	return amx, count
}

func (n *Net) mnistEval(image []byte, label GoMNIST.Label) {
	z := n.Eval(smash(image))
	if z[len(z)-1][int(label)] == 1 {
		met.WithLabelValues(fmt.Sprintf("mnist_easy_%d", label)).Observe(1.0)
	} else {
		met.WithLabelValues(fmt.Sprintf("mnist_easy_%d", label)).Observe(0.0)
	}

	if m, c := argmax(z[len(z)-1]); m == int(label) {
		met.WithLabelValues(fmt.Sprintf("mnist_hard_%d", label)).Observe(1.0 / float64(c))
	} else {
		met.WithLabelValues(fmt.Sprintf("mnist_hard_%d", label)).Observe(0.0)
	}
	//log.Println(x.Eval(smash(image)))
}

func (n *Net) Summarize() map[int]int {
	outp := make(map[int]int)
	for _, xx := range (*n).Layers {
		for _, yy := range xx.Gates {
			for _, zz := range yy {
				outp[zz] += 1
			}
		}
	}
	return outp
}

func main() {
	rand.Seed(time.Now().UnixNano())
	x := New([]int{784, 128, 10})
	x.Randomize()
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}

	size := 201
	inps := make([][]int, size, size)
	outs := make([][]int, size, size)
	sweeper := train.Sweep()
	g := 0.
	zeros := 0
	for ii := 0; ii < 60000*30; ii++ {
		if (ii > 0) && (ii%size == 0) {
			g = x.BProp(inps, outs, 5.0, 4.*math.Exp(-float64((zeros+1)*ii)/60000.0))
			if g == 0 {
				zeros++
			} else {
				zeros = 0
			}
			if zeros > 10 {
				break
			}
		}
		image, label, present := sweeper.Next()
		if !present {
			log.Println(ii, g, 4.*math.Exp(-float64((zeros+1)*ii)/60000.0))
			sweeper = train.Sweep()
			image, label, present = sweeper.Next()
		}
		inps[ii%size] = smash(image)
		outs[ii%size] = onehot(int(label), 10)
		//x.mnistEval(image, label)
	}
	log.Println(promdump(1))
	sweeper = test.Sweep()
	for ii := 0; ii < 10000; ii++ {
		image, label, present := sweeper.Next()
		if !present {
			break
		}
		x.mnistEval(image, label)
	}
	log.Println(promdump(1))
	log.Println(x.Summarize())

}
