Download Link: https://assignmentchef.com/product/solved-fundamentals-of-computer-vision-project-3-tracking-objects-in-videos
<br>
<h1 style="margin-bottom: 0.540791em; font-size: 2.61792em;">Overview</h1>

One incredibly important aspect of human and animal vision is the ability to follow objects and people in our view. Whether it is a tiger chasing its prey, or you trying to catch a basketball, tracking is so integral to our everyday lives that we forget how much we rely on it. In this assignment, you will be implementing an algorithm that will track an object in a video.

You will first implement the Lucas-Kanade tracker, and then a more computationally efficient version called the Matthew-Baker (or inverse compositional) method [1]. This method is one of the most commonly used methods in computer vision due to its simplicity and wide applicability. We have provided two video sequences: a car on a road, and a helicopter approaching a runway.

To initialize the tracker you need to define a template by drawing a bounding box around the object to be tracked in the first frame of the video. For each of the subsequent frames the tracker will update an affine transform that warps the current frame so that the template in the first frame is aligned with the warped current frame.

<h1>Preliminaries</h1>

An image transformation or warp is an operation that acts on pixel coordinates and maps pixel values from one place to another in an image. Translation, rotation and scaling are all examples of warps. We will use the symbol <strong>W </strong>to denote warps. A warp function <strong>W </strong>has a set of parameters <strong>p </strong>associated with it and maps a pixel with coordinates <strong>x </strong>= [<em>u v</em>]<em><sup>T </sup></em>to <strong>x</strong>0 = [<em>u</em>0 <em>v</em>0]<em>T</em>.

<strong>x</strong><sup>0 </sup>= <strong>W</strong>(<strong>x</strong>;<strong>p</strong>)                                                                 (1)

An affine transform is a warp that can include any combination of translation, anisotropic scaling and rotations. An affine warp can be parametrized in terms of 6 parameters <strong>p </strong>=

[<em>p</em><sub>1 </sub><em>p</em><sub>2 </sub><em>p</em><sub>3 </sub><em>p</em><sub>4 </sub><em>p</em><sub>5 </sub><em>p</em><sub>6</sub>]<em><sup>T</sup></em>. One of the convenient things about an affine transformation is that it is linear; its action on a point with coordinates <strong>x </strong>= [<em>u v</em>]<em><sup>T </sup></em>can be described as a matrix operation

(2)

Where <strong>W</strong>(<strong>p</strong>) is a 3 × 3 matrix such that

<strong>W</strong>                                                 (3)

Note that for convenience when we want to refer to the warp as a function we will use <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) and when we want to refer to the matrix for an affine warp we will use <strong>W</strong>(<strong>p</strong>). Table 1 contains a summary of the variables used in the next two sections. It will be useful to keep these in mind.

Table 1: Summary of Variables

<table width="520">

 <tbody>

  <tr>

   <td width="67">Symbol</td>

   <td width="147">Vector/Matrix Size</td>

   <td width="307">Description</td>

  </tr>

  <tr>

   <td width="67"><em>u</em></td>

   <td width="147">1 × 1</td>

   <td width="307">Image horizontal coordinate</td>

  </tr>

  <tr>

   <td width="67"><em>v</em></td>

   <td width="147">1 × 1</td>

   <td width="307">Image vertical coordinate</td>

  </tr>

  <tr>

   <td width="67"><strong>x</strong></td>

   <td width="147">2 × 1 or 1 × 1</td>

   <td width="307">pixel coordinates: (<em>u,v</em>) or unrolled</td>

  </tr>

  <tr>

   <td width="67"><strong>I</strong></td>

   <td width="147"><em>m </em>× 1</td>

   <td width="307">Image unrolled into a vector (<em>m </em>pixels)</td>

  </tr>

  <tr>

   <td width="67"><strong>T</strong></td>

   <td width="147"><em>m </em>× 1</td>

   <td width="307">Template unrolled into a vector (<em>m </em>pixels)</td>

  </tr>

  <tr>

   <td width="67"><strong>W</strong>(<strong>p</strong>)</td>

   <td width="147">3 × 3</td>

   <td width="307">Affine warp matrix</td>

  </tr>

  <tr>

   <td width="67"></td>

   <td width="147">6 × 1</td>

   <td width="307">parameters of affine warp</td>

  </tr>

  <tr>

   <td width="67"></td>

   <td width="147"><em>m </em>× 1</td>

   <td width="307">partial derivative of image wrt <em>u</em></td>

  </tr>

  <tr>

   <td width="67"><em>∂v</em></td>

   <td width="147"><em>m </em>× 1</td>

   <td width="307">partial derivative of image wrt <em>v</em></td>

  </tr>

  <tr>

   <td width="67"><em><u>∂</u></em><strong><u>T </u></strong><em>∂u</em></td>

   <td width="147"><em>m </em>× 1</td>

   <td width="307">partial derivative of template wrt <em>u</em></td>

  </tr>

  <tr>

   <td width="67"></td>

   <td width="147"><em>m </em>× 1</td>

   <td width="307">partial derivative of template wrt <em>v</em></td>

  </tr>

  <tr>

   <td width="67">∇<strong>I</strong></td>

   <td width="147"><em>m </em>× 2</td>

   <td width="307">image gradient</td>

  </tr>

  <tr>

   <td width="67">∇<strong>T</strong></td>

   <td width="147"><em>m </em>× 2</td>

   <td width="307">image gradient</td>

  </tr>

  <tr>

   <td width="67"><em><u>∂</u></em><strong><u>W</u></strong><em>∂</em><strong>p</strong></td>

   <td width="147">2 × 6</td>

   <td width="307">Jacobian of affine warp wrt its parameters</td>

  </tr>

  <tr>

   <td width="67"><strong>J</strong></td>

   <td width="147"><em>m </em>× 6</td>

   <td width="307">Jacobian of error function <em>L </em>wrt <strong>p</strong></td>

  </tr>

  <tr>

   <td width="67"><strong>H</strong></td>

   <td width="147">6 × 6</td>

   <td width="307">Pseudo Hessian of <em>L </em>wrt <strong>p</strong></td>

  </tr>

 </tbody>

</table>

<h2>Lucas-Kanade: Forward Additive Alignment</h2>

A Lucas Kanade tracker maintains a warp <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) which aligns a sequence of images <strong>I</strong><em><sub>t </sub></em>to a template <strong>T</strong>. We denote pixel locations by <strong>x</strong>, so <strong>I</strong>(<strong>x</strong>) is the pixel value at location <strong>x </strong>in image <strong>I</strong>. For the purposes of this derivation, <strong>I </strong>and <strong>T </strong>are treated as column vectors (think of them as unrolled image matrices). <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) is the point obtained by warping <strong>x </strong>with a transform that has parameters <strong>p</strong>. <strong>W </strong>can be any transformation that is continuous in its parameters <strong>p</strong>. Examples of valid warp classes for <strong>W </strong>include translations (2 parameters), affine transforms (6 parameters) and full projective transforms (8 parameters). The Lucas Kanade tracker minimizes the pixel-wise sum of square difference between the warped image <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)) and the template <strong>T</strong>.

In order to align an image or patch to a reference template, we seek to find the parameter vector <strong>p </strong>that minimizes <em>L</em>, where:

<em>L </em>= <sup>X</sup>[<strong>T</strong>(<strong>x</strong>) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>))]<sup>2                                                                                                            </sup>(4)

<strong>x</strong>

In general this is a difficult non-linear optimization, but if we assume we already have a close estimate <strong>p </strong>of the correct warp, then we can assume that a small linear change ∆<strong>p </strong>is enough to get the best alignment. This is the forward additive form of the warp. The objective can then be written as:

<table width="601">

 <tbody>

  <tr>

   <td width="440"><em>L </em>= <sup>X</sup>[<strong>T</strong>(<strong>x</strong>) − <em>I </em>(<strong>W</strong>(<strong>x</strong>;<strong>p </strong>+ ∆<strong>p</strong>))]<sup>2</sup><strong>x</strong>Expanding this to the first order with Taylor Series gives us:</td>

   <td width="160">(5)</td>

  </tr>

 </tbody>

</table>

<em>L </em>≈ <sup>X</sup>                                    (6)

<strong>x</strong>

Here , which is the vector containing the horizontal and vertical gradient at pixel location <strong>x</strong>. Rearranging the Taylor expansion, it can be rewritten as a typical least squares approximation ∆<strong>p</strong><sup>∗ </sup>= argmin||<em>A</em>∆<em>p </em>− <em>b</em>||<sup>2</sup>

∆<em>p</em>

2

∆<strong>p</strong><sup>∗ </sup>= argmin                                                                                              (7)

<strong>x</strong>

This can be solved with ∆<strong>p</strong><sup>∗ </sup>= (<em>A<sup>T</sup>A</em>)<sup>−1</sup><em>A<sup>T</sup>b </em>where:

(<em>A<sup>T</sup>A</em>) = <strong>H </strong>= <sup>X</sup>                                             (8)

<strong>x</strong>

<em>A </em>= <sup>X</sup>                                                                (9)

<strong>x</strong>

<em>b </em>= <strong>T</strong>(<strong>x</strong>) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>))                                                  (10)

Once ∆<strong>p </strong>is computed, the best estimate warp can be updated <strong>p </strong>← <strong>p </strong>+ ∆<strong>p</strong>, and the whole procedure can be repeated again, stopping when ∆<strong>p </strong>is less than some threshold.

<h2>Matthew-Baker: Inverse Compositional Alignment</h2>

While Lucas-Kanade alignment works very well, it is computationally expensive. The inverse compositional method is similar, but requires less computation, as the Hessian and Jacobian only need to be computed once. One caveat is that the warp needs to be invertible. Since affine warps are invertible, we can use this method.

In the previous section, we combined two warps by simply adding one parameter vector to another parameter vector, and produce a new warp <strong>W</strong>(<strong>x</strong><em>,</em><strong>p</strong>+<strong>p</strong><sup>0</sup>). Another way of combining warps is through composition of warps. After applying a warp <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) to an image, another warp <strong>W</strong>(<strong>x</strong>;<strong>q</strong>) can be applied to the warped image. The resultant (combined) warp is

<strong>W</strong>(<strong>x</strong>;<strong>q</strong>) ◦ <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) = <strong>W</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)<em>,</em><strong>q</strong>)                                            (11)

Since affine warps can be implemented as matrix multiplications, composing two affine warps reduces to multiplying their corresponding matrices

<strong>W</strong>(<strong>x</strong>;<strong>q</strong>) ◦ <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) = <strong>W</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)<em>,</em><strong>q</strong>) = <strong>W</strong>(<strong>W</strong>(<strong>p</strong>)<strong>x</strong><em>,</em><strong>q</strong>) = <strong>W</strong>(<strong>q</strong>)<strong>W</strong>(<strong>p</strong>)<strong>x                   </strong>(12)

An affine transform can also be inverted. The inverse warp of <strong>W</strong>(<strong>p</strong>) is simply the matrix inverse of <strong>W</strong>(<strong>p</strong>), <strong>W</strong>(<strong>p</strong>)<sup>−1</sup>. In this assignment it will sometimes be simpler to consider an affine warp as a set of 6 parameters in a vector <strong>p </strong>and it will sometimes be easier to work with the matrix version <strong>W</strong>(<strong>p</strong>). Fortunately, switching between these two forms is easy (Equation 3).

The minimization is performed using an iterative procedure by making a small change (∆<strong>p</strong>) to <strong>p </strong>at each iteration. It is computationally more efficient to do the minimization by finding the ∆<strong>p </strong>that helps align the template to the image, than applying the inverse warp to the image. This is because the image will change with each frame of the video, but the template is fixed at initialization. We will see soon that doing this allows us to write the Hessian and Jacobian in terms of the template, and so this can be computed once at the beginning of the tracking. Hence at each step, we want to find the ∆<strong>p </strong>to minimize

<em>L </em>= <sup>X</sup>[<strong>T</strong>(<strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>)) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>))]<sup>2                                                                                           </sup>(13)

<strong>x</strong>

For tracking a patch template, the summation is performed only over the pixels lying inside the template region. We can expand <strong>T</strong>(<strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>)) in terms of its first order linear approximation to get

<em>L </em>≈ <sup>X</sup>                                 (14)

<strong>x</strong>

Where ∇<strong>T</strong>(<strong>x</strong>) = h<em><sup>∂</sup></em><strong><sup>T</sup></strong><em><sub>∂u</sub></em><sup>(<strong>x</strong>) <em>∂</em><strong>T</strong></sup><em><sub>∂v</sub></em><sup>(<strong>x</strong>)</sup><sup>i</sup>. To minimize we need to take the derivative of <em>L </em>and set it to zero

<em>∂L</em>

(15)

<strong>x</strong>

Setting to zero, switching from summation to vector notation and solving for ∆<strong>p </strong>we get

∆<strong>p </strong>= <strong>H</strong><sup>−1</sup><strong>J</strong><em><sup>T </sup></em>[<strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)) − <strong>T</strong>]                                                  (16)

where <strong>J </strong>is the Jacobian of <strong>T</strong>(<strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>)), <strong>J </strong>= ∇<strong>T</strong><em><u><sup>∂</sup></u><sub>∂</sub></em><strong><u><sup>W</sup></u><sub>p </sub></strong>, <strong>H </strong>is the approximated Hessian

<strong>H </strong>= <strong>J</strong><em><sup>T</sup></em><strong>J </strong>and <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)) is the warped image. Note that for a given template, the Jacobian <strong>J </strong>and Hessian <strong>H </strong>are independent of <strong>p</strong>. This means they only need to be computed once and then they can be reused during the entire tracking sequence.

Once ∆<strong>p </strong>has been solved for, it needs to be inverted and composed with <strong>p </strong>to get the new warp parameters for the next iteration.

<strong>W</strong>(<strong>x</strong>;<strong>p</strong>) ← <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) ◦ <strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>)<sup>−1                                                                                            </sup>(17)

The next iteration solves Equation 16 starting with the new value of <strong>p</strong>. Possible termination criteria include the absolute value of ∆<strong>p </strong>falling below some value or running for some fixed number of iterations.

<h1>1       Theory Questions                                                         (25 points)</h1>

<strong>Type down your answers for the following questions in your write-up. </strong>Each question should only take a couple of lines. In particular, the “proofs” do not require any lengthy calculations. If you are lost in many lines of complicated algebra you are doing something much too complicated (or wrong).

<h2>Q1.1: Calculating the Jacobian                                                                                     (15 points)</h2>

Assuming the affine warp model defined in Equation 3, derive the expression for the Jacobian Matrix <strong>J </strong>in terms of the warp parameters <strong>p </strong>= [<em>p</em><sub>1 </sub><em>p</em><sub>2 </sub><em>p</em><sub>3 </sub><em>p</em><sub>4 </sub><em>p</em><sub>5 </sub><em>p</em><sub>6</sub>]<sup>0</sup>.

<h2>Q1.2: Computational complexity                                                                                (10 points)</h2>

Find the computational complexity (Big O notation) for the initialization step (Precomputing <strong>J </strong>and <strong>H</strong><sup>−1</sup>) and for each runtime iteration (Equation 16) of the Inverse Compositional method. Express your answers in terms of <em>n</em>, <em>m </em>and <em>p </em>where <em>n </em>is the number of pixels in the template <strong>T</strong>, <em>m </em>is the number of pixels in an input image <strong>I </strong>and <em>p </em>is the number of parameters used to describe the warp <em>W</em>. How does this compare to the run time of the regular Lucas-Kanade method?

<h1>2       Lucas-Kanade Tracker                                              (60 points)</h1>

For this section, TA  will grade your tracker based on the performance you achieved on the two provided video sequences: (1) data/car1/. The provided script files lk demo.m and mb demo.m handle reading in images, t<u>e</u>mplate region marking, making tracker function calls and displaying output onto the screen. The function prototypes provided are guidelines. Please make sure that your code runs functionally with the original script and generates the outputs we are looking for (a frame sequence with the bounding box of the target being tracked on each frame) so that we can replicate your results.

Note that the only thing TA  would do for you during grading is change the input data directory, and initialize your tracker based on what you mentioned in your write-up. Please submit one video for each of them in the results/ directory, with file name car.mp4. Also, please mention the initialization coordinates of your tracker for both video sequences in your write-up and in your code.

<h2>Q2.1: Write a Lucas-Kanade Tracker for a Flow Warp                                           (20 points)</h2>

Write the function with the following function signature:

[u,v] = LucasKanade(It, It1, rect) that computes the optimal local motion from frame <strong>I</strong><em><sub>t </sub></em>to frame <strong>I</strong><em><sub>t</sub></em><sub>+1 </sub>that minimizes Equation 1. Here It is the image frame <strong>I</strong><em><sub>t</sub></em>, It1 is the image frame <strong>I</strong><em><sub>t</sub></em><sub>+1</sub>, and rect is the 4×1 vector that represents a rectangle on the image frame It. The four components of the rectangle are [x, y, w, h], where (x, y) is the top-left corner and (w, h) is the width and height of the bounding box. The rectangle is inclusive, i.e., in includes all the four corners. To deal with fractional movement of the template, you will need to interpolate the image using the Matlab function interp2. You will also need to iterate the estimation until the change in warp parameters (u, v) is below a threshold. Use the forward compositional (Lucas-Kanade method) for this question.

<h2>Q2.2: Initializing the Matthew-Baker Tracker                                                         (10 points)</h2>

Write the function initAffineMBTracker() that initializes the inverse compositional tracker by precomputing important matrices needed to track a template patch.

function [affineMBContext] = initAffineMBTracker(img, rect)

The function will input a greyscale image (img) along with a bounding box (rect) (in the format [x y w h]).

The function should output a Matlab structure affineMBContext that contains the Jacobian of the affine warp with respect to the 6 affine warp parameters and the inverse of the approximated Hessian matrix (<strong>J </strong>and <strong>H</strong><sup>−1 </sup>in Equation 16).

<h2>Q2.3: The Main Matthew-Baker Tracker                                                                  (20 points)</h2>

Write the function affineMBTracker() that does the actual template tracking. function [Wout] = affineMBTracker(img, tmp, rect, Win, context) The function will input a greyscale image of the current frame (img), the template image (tmp), the bounding box rect that marks the template region in tmp, The affine warp matrix for the previous frame (Win) and the precomputed <strong>J </strong>and <strong>H</strong><sup>−1 </sup>matrices context.

The function should output the 3 × 3 matrix Wout that contains the new affine warp matrix updated so that it aligns the current frame with the template.

You can either used a fixed number of gradient descent iterations or formulate a stopping criteria for the algorithm. You can use the included image warping function to apply affine warps to images.

<h2>Q2.4: Tracking a Car                                                                                                      (10 points)</h2>

Test your trackers on the short car video sequence (data/car1/) by running the wrapper scripts lk demo.m and mb demo.m. What sort of templates work well for tracking? At what point does the tracker break down? Why does this happen?

<strong>In your write-up: </strong>Submit your best video of the car being tracked.         Save it as results/car.mp4.

Figure 2: Tracking in the car image sequences




<h1>4        Submission Summary</h1>

<ul>

 <li><strong>1 </strong>Derive the expression for the Jacobian Matrix</li>

 <li><strong>2 </strong>What is the computational complexity of inverse compositional method?</li>

 <li><strong>1 </strong>Write the forward compositional tracker (LK Tracker)</li>

 <li><strong>2 </strong>Initialize the inverse compositional tracker (MB Tracker)</li>

 <li><strong>3 </strong>Write the inverse compositional tracker (MB Tracker)</li>

 <li><strong>4 </strong>Run the inverse compositional tracker on the car dataset. What templates does it work well with? When does the tracker break down? Why does this happen?</li>

 <li><strong>5</strong> Run the inverse compositional tracker on the run markings dataset.</li>

</ul>

<h1>References</h1>

<ul>

 <li>Simon Baker, et al. Lucas-Kanade 20 Years On: A Unifying Framework: Part 1, CMURI-TR-02-16, Robotics Institute, Carnegie Mellon University, 2002</li>

 <li>Simon Baker, et al. Lucas-Kanade 20 Years On: A Unifying Framework: Part 2, CMURI-TR-03-35, Robotics Institute, Carnegie Mellon University, 2003</li>

 <li>Bouguet, Jean-Yves. Pyramidal Implementation of the Lucas Kanade Feature Tracker:</li>

</ul>

Description of the algorithm, Intel Corporation, 2001

<h2>Credit</h2>

This project is adapted directly from Ioannis Gkioulekas.

12