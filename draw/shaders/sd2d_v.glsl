attribute vec2 pos;
varying vec2 v_pos;
uniform vec2 screen_size;
uniform mat2 matrix;

void main()
{
  v_pos = pos*screen_size/2;
  v_pos = matrix*v_pos;
  gl_Position = vec4(pos, 0.0, 1.0);
}
