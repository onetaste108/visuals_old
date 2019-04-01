varying vec2 t_pos;
varying vec2 v_pos;
uniform vec2 viewport;
sampler2D tex;
sampler2D warp;
void main() {
  vec2 off = texture2D(warp,t_pos).xy;
  vec2 sig = texture2D(warp,t_pos).ba;
  gl_FragColor = texture2D(tex,(t_pos+off*sig));
}
