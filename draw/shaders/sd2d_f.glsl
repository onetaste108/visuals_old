// COLORS ---------------------------------------------------------------------

vec4 fill(float d, vec4 c, float aa)
{
    float a = mix(0.0, c.a, 1-smoothstep(0.0,aa,d));
    return vec4(c.rgb, a);
}

vec4 stroke(float d, vec4 c, float aa, float w)
{
    float a = mix(0.0, c.a, 1-smoothstep(w/2,w/2+aa,abs(d)));
    return vec4(c.rgb, a);
}

// SHAPES ---------------------------------------------------------------------

float circle(vec2 p, vec2 c, float r)
{
  return length(p-c)-r;
}

float line(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p-a, ba = b-a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h );
}

// VARIABLES ------------------------------------------------------------------

varying vec2 v_pos;
uniform vec2 ih;
uniform vec2 jh;

uniform vec4 color;
uniform float stroke_weight;
uniform float aa;

uniform vec2 center;
uniform float radius;

uniform vec2 p1;
uniform vec2 p2;

uniform int COL_MODE;
#define FILL 0
#define STROKE 1

uniform int SHAPE;
#define CIRCLE 0
#define LINE 1


// MAIN -----------------------------------------------------------------------

void main()
{

  vec2 pos = v_pos;

  float d;
  if (SHAPE == CIRCLE)
  {
    d = circle(pos, center, radius);
  }
  else if (SHAPE == LINE)
  {
    vec2 np1 = p1[0]*ih + p1[1]*jh;
    vec2 np2 = p2[0]*ih + p2[1]*jh;
    d = line(pos, np1, np2);
  }

  vec4 c;
  if (COL_MODE == FILL)
  {
    c = fill(d, color, aa);
  }
  else if (COL_MODE == STROKE)
  {
    c = stroke(d, color, aa, stroke_weight);
  }

  gl_FragColor = c;
}
