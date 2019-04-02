var canvas = document.getElementById("c");
var gl = canvas.getContext("webgl");
if (!gl) alert("No GL for you!");
else console.log("GL loaded!");



var vss = `
attribute vec4 pos;
varying vec2 v_pos;
void main() {
  v_pos = pos.xy;
  gl_Position = pos;
}
`;
var fss = `
precision mediump float;
varying vec2 v_pos;
void main() {
  gl_FragColor = vec4(sin(v_pos.x*20.0)/2.0+0.5,0,0,1);
}
`;

var vs = createShader(gl, gl.VERTEX_SHADER, vss);
var fs = createShader(gl, gl.FRAGMENT_SHADER, fss);

var prog = createProgram(gl, vs, fs);

var pos_loc = gl.getAttribLocation(prog, "pos");
var pos_buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, pos_buf);

var positions = [
  -1,  -1,
  -1,  1,
  1,  -1,
  1,  -1,
  -1, 1,
  1,  1
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

var drawtype = gl.TRIANGLES;
var offset = 0;
var count = 6;
gl.drawArrays(drawtype, offset, count);

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
