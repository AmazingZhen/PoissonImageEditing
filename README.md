# PoissonImageEditing
Reproducd famous paper <b>"Poisson Image Editing"</b> in SIGGRAPH 2003.  
This paper is very useful for <b>copying and pasting between images</b>.  
Based on OpenCV(2.4.9) and Eigen.  
Developed under Visual Stdio 2015, WIN 10 64 bits.  

## Input
- Two same size image. A source and a destination.
- A mask of editing region (if no mask exists, the program can generate one).
### src
- ![src](https://github.com/AmazingZhen/PoissonImageEditing/blob/master/PoissonImageEditing/input/src.jpg?raw=true)
### dst
- ![dst](https://github.com/AmazingZhen/PoissonImageEditing/blob/master/PoissonImageEditing/input/dst.jpg?raw=true)
### mask
- ![dst](https://github.com/AmazingZhen/PoissonImageEditing/blob/master/PoissonImageEditing/input/mask.png?raw=true)
  
## Output
- An image which clones masked region of source image to destination image.
### No mixing
- ![result1](https://github.com/AmazingZhen/PoissonImageEditing/blob/master/PoissonImageEditing/res/seamless_cloning_res.jpg?raw=true)
### Mixing
- ![result2](https://github.com/AmazingZhen/PoissonImageEditing/blob/master/PoissonImageEditing/res/mixed_seamless_cloning_res.jpg?raw=true)
  
## Algorithmic process
// TO DO ...
  
## Areas for improvement
- Actually it's no need to restrict two images in same size, but it's not the key of algorithm.
