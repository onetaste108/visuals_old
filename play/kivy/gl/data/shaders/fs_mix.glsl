varying vec2 t_pos;
varying vec2 v_pos;

uniform sampler2D tex1;
uniform sampler2D tex2;

void main() {
  gl_FragColor = texture2D(tex1,t_pos)+texture2D(tex2,t_pos);
}
