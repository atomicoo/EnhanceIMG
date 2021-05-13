# Convolution arithmetic

A technical report on convolution arithmetic in the context of deep learning.

The code and the images of this tutorial are free to use as regulated by the 
licence and subject to proper attribution:

* \[1\] Vincent Dumoulin, Francesco Visin - [A guide to convolution arithmetic
  for deep learning](https://arxiv.org/abs/1603.07285)
  ([BibTeX](https://gist.github.com/fvisin/165ca9935392fa9600a6c94664a01214))

## Convolution animations

_N.B.: Blue maps are inputs, and cyan maps are outputs._

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902094-no_padding_no_strides.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902157-arbitrary_padding_no_strides.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902207-same_padding_no_strides.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902239-full_padding_no_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Half padding, no strides</td>
    <td>Full padding, no strides</td>
  </tr>
  <tr>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902287-no_padding_strides.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902345-padding_strides.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902378-padding_strides_odd.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides</td>
    <td>Padding, strides</td>
    <td>Padding, strides (odd)</td>
    <td></td>
  </tr>
</table>


## Transposed convolution animations

_N.B.: Blue maps are inputs, and cyan maps are outputs._

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902426-no_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902476-arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902516-same_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902551-full_padding_no_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>Half padding, no strides, transposed</td>
    <td>Full padding, no strides, transposed</td>
  </tr>
  <tr>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902588-no_padding_strides_transposed.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902631-padding_strides_transposed.gif"></td>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902694-padding_strides_odd_transposed.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides, transposed</td>
    <td>Padding, strides, transposed</td>
    <td>Padding, strides, transposed (odd)</td>
    <td></td>
  </tr>
</table>


## Dilated convolution animations

_N.B.: Blue maps are inputs, and cyan maps are outputs._

<table style="width:25%"; table-layout:fixed;>
  <tr>
    <td><img width="150px" src="https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620902735-dilation.gif"></td>
  </tr>
  <tr>
    <td>No padding, no stride, dilation</td>
  </tr>
</table>


## References

[Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

