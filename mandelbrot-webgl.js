// WebGL-accelerated Mandelbrot renderer
class MandelbrotWebGL {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = null;
        this.program = null;
        this.uniforms = {};
        this.quadBuffer = null;
        
        this.init();
    }
    
    init() {
        // Get WebGL context
        this.gl = this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl');
        if (!this.gl) {
            throw new Error('WebGL not supported');
        }
        
        const gl = this.gl;
        
        // Vertex shader - simple quad covering the viewport
        const vertexShaderSource = `
            attribute vec2 a_position;
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;
        
        // Fragment shader - Mandelbrot calculation
        const fragmentShaderSource = `
            precision highp float;
            
            uniform vec2 u_resolution;
            uniform vec4 u_bounds; // xmin, xmax, ymin, ymax
            uniform int u_maxIterations;
            uniform float u_colorOffset;
            
            vec3 hslToRgb(float h, float s, float l) {
                h = mod(h, 360.0) / 360.0;
                s = s / 100.0;
                l = l / 100.0;
                
                float c = (1.0 - abs(2.0 * l - 1.0)) * s;
                float x = c * (1.0 - abs(mod(h * 6.0, 2.0) - 1.0));
                float m = l - c / 2.0;
                
                vec3 rgb;
                if (h < 1.0/6.0) {
                    rgb = vec3(c, x, 0.0);
                } else if (h < 2.0/6.0) {
                    rgb = vec3(x, c, 0.0);
                } else if (h < 3.0/6.0) {
                    rgb = vec3(0.0, c, x);
                } else if (h < 4.0/6.0) {
                    rgb = vec3(0.0, x, c);
                } else if (h < 5.0/6.0) {
                    rgb = vec3(x, 0.0, c);
                } else {
                    rgb = vec3(c, 0.0, x);
                }
                
                return rgb + m;
            }
            
            void main() {
                // Map pixel to complex plane
                vec2 uv = gl_FragCoord.xy / u_resolution;
                float x0 = mix(u_bounds[0], u_bounds[1], uv.x);
                float y0 = mix(u_bounds[2], u_bounds[3], 1.0 - uv.y); // Flip Y
                
                // Mandelbrot iteration
                float x = 0.0;
                float y = 0.0;
                int iteration = 0;
                
                for(int i = 0; i < 2000; i++) {
                    if(i >= u_maxIterations) break;
                    
                    float x2 = x * x;
                    float y2 = y * y;
                    
                    if(x2 + y2 > 4.0) {
                        iteration = i;
                        break;
                    }
                    
                    float xtemp = x2 - y2 + x0;
                    y = 2.0 * x * y + y0;
                    x = xtemp;
                    
                    iteration = i;
                }
                
                // Color the pixel
                if(iteration == u_maxIterations - 1) {
                    // Inside the set - black
                    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    // Outside - use smooth coloring
                    float smooth_iter = float(iteration);
                    if(x * x + y * y > 4.0) {
                        // Add fractional part for smooth coloring
                        float log_zn = log(x * x + y * y) / 2.0;
                        float nu = log(log_zn / log(2.0)) / log(2.0);
                        smooth_iter = float(iteration) + 1.0 - nu;
                    }
                    
                    float hue = mod(smooth_iter * 10.0 + u_colorOffset, 360.0);
                    vec3 color = hslToRgb(hue, 100.0, 50.0);
                    gl_FragColor = vec4(color, 1.0);
                }
            }
        `;
        
        // Compile shaders
        const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
        const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource);
        
        // Create program
        this.program = gl.createProgram();
        gl.attachShader(this.program, vertexShader);
        gl.attachShader(this.program, fragmentShader);
        gl.linkProgram(this.program);
        
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            throw new Error('Unable to link program: ' + gl.getProgramInfoLog(this.program));
        }
        
        // Get uniform locations
        this.uniforms = {
            resolution: gl.getUniformLocation(this.program, 'u_resolution'),
            bounds: gl.getUniformLocation(this.program, 'u_bounds'),
            maxIterations: gl.getUniformLocation(this.program, 'u_maxIterations'),
            colorOffset: gl.getUniformLocation(this.program, 'u_colorOffset')
        };
        
        // Create quad buffer (two triangles covering the viewport)
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1
        ]);
        
        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        
        // Get attribute location
        const positionLocation = gl.getAttribLocation(this.program, 'a_position');
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
    }
    
    compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            throw new Error('Shader compile error: ' + gl.getShaderInfoLog(shader));
        }
        
        return shader;
    }
    
    render(xmin, xmax, ymin, ymax, maxIterations = 100) {
        const gl = this.gl;
        
        // Update canvas size if needed
        const displayWidth = this.canvas.clientWidth;
        const displayHeight = this.canvas.clientHeight;
        
        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
        }
        
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        
        // Use program
        gl.useProgram(this.program);
        
        // Set uniforms
        gl.uniform2f(this.uniforms.resolution, this.canvas.width, this.canvas.height);
        gl.uniform4f(this.uniforms.bounds, xmin, xmax, ymin, ymax);
        gl.uniform1i(this.uniforms.maxIterations, maxIterations);
        gl.uniform1f(this.uniforms.colorOffset, 0);
        
        // Draw
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    
    // Get pixel data (for compatibility with existing code)
    getImageData() {
        const gl = this.gl;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        const pixels = new Uint8Array(width * height * 4);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        
        // Flip vertically (WebGL renders upside down)
        const imageData = new ImageData(width, height);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcIdx = ((height - 1 - y) * width + x) * 4;
                const dstIdx = (y * width + x) * 4;
                imageData.data[dstIdx] = pixels[srcIdx];
                imageData.data[dstIdx + 1] = pixels[srcIdx + 1];
                imageData.data[dstIdx + 2] = pixels[srcIdx + 2];
                imageData.data[dstIdx + 3] = pixels[srcIdx + 3];
            }
        }
        
        return imageData;
    }
}

// Export for use in main file
window.MandelbrotWebGL = MandelbrotWebGL;