varying vec2 t_pos;
varying vec2 v_pos;

uniform vec2 touch_begin;
uniform vec2 touch_end;
uniform vec2 viewport;
uniform int mode;

void main() {
  vec2 off;
  vec2 dir = touch_end-touch_begin;
  vec2 idir = vec2(-dir.y, dir.x);
  vec2 border = normalize(vec2(idir.y, -idir.x));
  float dist = dot(border, v_pos-touch_begin);
  vec2 sig;

  if (dist > 0) {
    off = abs(dir);
    sig = vec2(sign(dir.x),sign(dir.y));
  } else {
    off = vec2(0,0);
  }

  if (mode == 0) {
    off = vec2(0,0);
  }

  gl_FragColor = vec4(off/viewport,sig);
}
