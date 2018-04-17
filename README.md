# CartPoleLinearApproximation
First q learning. It failed maybe because theta times w_3 cannot express the lifespan, which is the sum of the future rewards.
The lifespan is longer when the theta is small number than when it is big number, but all this model can do is just to multiply it by w_3.<br>

lifespan(theta) = theta * w_3 > 0<br>

lifespan(1) = 1 * w_3<br>
  => must be some big positive number.<br>
  => w_3 must be big.<br>

lifespan(100) = 100 * w_3<br>
  => must be some small positive number.<br>
  => w_3 must be small.<br>
  => contradictory<br>

I think that's why this way does not work well, and the w diverges.<br>
