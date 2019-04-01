attribute vec2 vPosition;
uniform vec2 viewport;
varying vec2 t_pos;
varying vec2 v_pos;
void main() {
  t_pos = vPosition/2+0.5;
  v_pos = t_pos * viewport;
  gl_Position = vec4(vPosition,0,1);
}
