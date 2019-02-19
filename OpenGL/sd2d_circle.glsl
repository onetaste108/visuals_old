varying vec2 v_position;
uniform vec4 color;
uniform bool is_stroke;
uniform float stroke_weight;

uniform vec2 center;
uniform float radius;

vec4 fill(float d, vec4 c)
{
    vec4 none  = vec4(c.rgb, 0.0);
    vec4 color = mix(none, c, 1-smoothstep(0.0,0.01,d));
    return color;
}

vec4 stroke(float d, float w, vec4 c)
{
    vec4 none  = vec4(c.rgb, 0.0);
    vec4 color = mix(none, c, 1-smoothstep(stroke_weight/2,stroke_weight/2+0.01,abs(d)));
    return color;
}

float sd2d_circle(vec2 p, vec2 c, float radius)
{
  return length(p-c)-radius;
}

void main() {
  float d = sd2d_circle(v_position, center, radius);
  vec4 c;
  if (!is_stroke)
  {
    c = fill(d, color);
  }
  else
  {
    c = stroke(d, stroke_weight, color);
  }
  gl_FragColor = color;
}
