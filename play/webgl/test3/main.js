var canvas = document.getElementById("c");
var gl = canvas.getContext("webgl");
if (!gl) alert("No GL for you!");
else console.log("GL loaded!");



var vss = `
attribute vec2 pos;
uniform vec2 res;
varying vec2 v_pos;
void main() {
  v_pos = pos/res;
  v_pos = v_pos * 2.0;
  v_pos = v_pos - 1.0;
  v_pos = v_pos * vec2(1.0, -1.0);
  gl_Position = vec4(v_pos,0,1);
}
`;
var fss = `
precision mediump float;
uniform vec4 color;
varying vec2 v_pos;
void main() {
  gl_FragColor = color;
}
`;

var vs = createShader(gl, gl.VERTEX_SHADER, vss);
var fs = createShader(gl, gl.FRAGMENT_SHADER, fss);

var prog = createProgram(gl, vs, fs);

var pos_loc = gl.getAttribLocation(prog, "pos");
var pos_buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, pos_buf);

var res_loc = gl.getUniformLocation(prog, "res");
var col_loc = gl.getUniformLocation(prog, "color");

var positions = [
  10,10,
  10,100,
  100,10,
  100,10,
  10,100,
  100,100
]
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);


resizeCanvas(canvas);
gl.viewport(0,0,canvas.width,canvas.height);
gl.clearColor(0,0,0,1);
gl.clear(gl.COLOR_BUFFER_BIT);
gl.useProgram(prog);


gl.enableVertexAttribArray(pos_loc);
gl.bindBuffer(gl.ARRAY_BUFFER, pos_buf);

var size = 2;
var type = gl.FLOAT;
var normalize = false;
var stride = 0;
var offset = 0;
gl.vertexAttribPointer(pos_loc, size, type, normalize, stride, offset);

gl.uniform2f(res_loc, gl.canvas.width, gl.canvas.height);

for (var ii = 0; ii < 50; ++ii) {
  setRect(gl, random(300), random(300), random(300), random(300));
  gl.uniform4f(col_loc, Math.random(), Math.random(), Math.random(), Math.random());
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function createShader(gl, type, source) {
  var shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (success) return shader;
  console.log(gl.getShaderInfoLog(shader));
  gl.deleteShader(shader);
}

function createProgram(gl, vs, fs) {
  var program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  var success = gl.getProgramParameter(program, gl.LINK_STATUS);
  if (success) return program;
  console.log(gl.getProgramInfoLog(program));
  gl.deleteProgram(program);
}

function resizeCanvas(canvas) {
  var displayWidth = canvas.clientWidth;
  var displayHeight = canvas.clientHeight;
  if (canvas.width != displayWidth || canvas.height != displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
  }
}

function random(range) {
  return Math.floor(Math.random() * range);
}

function setRect(gl, x, y, w, h) {
  var x1 = x;
  var x2 = x + w;
  var y1 = y;
  var y2 = y + h;

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    x1, y1,
    x1, y2,
    x2, y1,
    x2, y1,
    x1, y2,
    x2, y2
  ]), gl.STATIC_DRAW);
}
