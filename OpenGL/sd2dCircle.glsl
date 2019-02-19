varying vec2 v_position;
uniform bool is_stroke;
uniform float stroke_weight;

uniform vec2 center;
uniform float radius;

vec4 fill(float d)
{
    vec4 white = vec4(1.0, 1.0, 1.0, 1.0);
    vec4 none  = vec4(1.0, 1.0, 1.0, 0.0);
    vec4 color = mix(none, white, 1-smoothstep(0.0,0.01,d));
    return color;
}

vec4 stroke(float d)
{
    vec4 red   = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 none  = vec4(1.0, 0.0, 0.0, 0.0);
    vec4 color = mix(none, red, 1-smoothstep(stroke_weight/2,stroke_weight/2+0.01,abs(d)));
    return color;
}

float sd2d_circle(vec2 p, vec2 c, float radius)
{
  return length(p-c)-radius;
}

void main() {
  float d = sd2d_circle(v_position, center, radius);
  vec4 color;
  if is_fill {
    color = fill(d);
  }
  if is_stroke {
    color = stroke(d);
  }
  gl_FragColor = color;
}
