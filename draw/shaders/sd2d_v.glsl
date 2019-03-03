attribute vec2 pos;
varying vec2 v_pos;
uniform vec2 screen_size;

void main()
{
  v_pos = pos*screen_size/2;
  gl_Position = vec4(pos, 0.0, 1.0);
}
