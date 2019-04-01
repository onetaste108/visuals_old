varying vec2 t_pos;
varying vec2 v_pos;

sampler2D tex;
void main() {
  gl_FragColor = texture2D(tex,t_pos);
}
