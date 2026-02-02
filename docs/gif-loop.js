/**
 * Doc animated GIFs with a frame scrubber slider.
 * Add class="doc-animated-gif" and data-frames to img.
 * Loads GIF, decodes frames, and shows a slider to scrub through them.
 */
(function() {
    var style = document.createElement('style');
    style.textContent = `
        .gif-container { display: inline-block; position: relative; max-width: 100%; }
        .gif-container canvas { max-width: 100%; height: auto; display: block; }
        .gif-slider-wrap { margin-top: 8px; padding: 8px; background: #f5f5f5; border-radius: 4px; }
        .gif-slider-wrap label { display: block; font-size: 12px; color: #555; margin-bottom: 4px; }
        .gif-slider-wrap input[type="range"] { width: 100%; cursor: pointer; }
    `;
    document.head.appendChild(style);

    function compositeFrames(frames, gifWidth, gifHeight) {
        var fullFrames = [];
        var prevImage = null;
        var disposalTypes = [1, 2, 3]; // 1=clear to bg, 2=leave, 3=restore prev
        for (var i = 0; i < frames.length; i++) {
            var f = frames[i];
            var dims = f.dims;
            var w = gifWidth;
            var h = gifHeight;
            var canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            var ctx = canvas.getContext('2d');
            if (i === 0) {
                ctx.fillStyle = '#fff';
                ctx.fillRect(0, 0, w, h);
            } else {
                var prev = fullFrames[i - 1];
                var disp = f.disposalType !== undefined ? f.disposalType : 2;
                if (disp === 2) {
                    ctx.drawImage(prev, 0, 0);
                } else if (disp === 3 && prevImage) {
                    ctx.putImageData(prevImage, 0, 0);
                } else {
                    ctx.fillStyle = '#fff';
                    ctx.fillRect(0, 0, w, h);
                    if (prev) ctx.drawImage(prev, 0, 0);
                }
            }
            if (f.patch) {
                var patchData = ctx.createImageData(dims.width, dims.height);
                patchData.data.set(f.patch);
                ctx.putImageData(patchData, dims.left, dims.top);
            }
            prevImage = ctx.getImageData(0, 0, w, h);
            fullFrames.push(canvas);
        }
        return fullFrames;
    }

    function initOne(img, parseGIF, decompressFrames) {
        var framesAttr = img.getAttribute('data-frames');
        var numFrames = framesAttr ? parseInt(framesAttr, 10) : 0;
        if (!numFrames) return;

        var url = img.src.split('?')[0];
        fetch(url)
            .then(function(r) { return r.arrayBuffer(); })
            .then(function(buffer) {
                var gif = parseGIF(new Uint8Array(buffer));
                var frames = decompressFrames(gif, true);
                if (!frames.length) return;

                var w = gif.lsd.width;
                var h = gif.lsd.height;
                var fullFrames = compositeFrames(frames, w, h);

                var container = document.createElement('div');
                container.className = 'gif-container';
                img.parentNode.insertBefore(container, img);
                img.style.display = 'none';

                var canvas = document.createElement('canvas');
                canvas.width = w;
                canvas.height = h;
                canvas.style.width = img.style.width || '100%';
                canvas.style.maxWidth = img.style.maxWidth || '100%';
                container.appendChild(canvas);

                var sliderWrap = document.createElement('div');
                sliderWrap.className = 'gif-slider-wrap';
                var label = document.createElement('label');
                label.textContent = 'Frame 0 / ' + (fullFrames.length - 1);
                var range = document.createElement('input');
                range.type = 'range';
                range.min = 0;
                range.max = Math.max(0, fullFrames.length - 1);
                range.value = 0;
                sliderWrap.appendChild(label);
                sliderWrap.appendChild(range);
                container.appendChild(sliderWrap);

                function showFrame(idx) {
                    idx = Math.max(0, Math.min(idx, fullFrames.length - 1));
                    var ctx = canvas.getContext('2d');
                    ctx.drawImage(fullFrames[idx], 0, 0);
                    label.textContent = 'Frame ' + idx + ' / ' + (fullFrames.length - 1);
                }

                showFrame(0);
                range.addEventListener('input', function() {
                    showFrame(parseInt(range.value, 10));
                });
            })
            .catch(function() {
                img.style.display = '';
            });
    }

    function init() {
        var imgs = document.querySelectorAll('img.doc-animated-gif');
        if (!imgs.length) return;
        import('https://esm.sh/gifuct-js@2.1.2').then(function(mod) {
            var parseGIF = mod.parseGIF;
            var decompressFrames = mod.decompressFrames;
            if (parseGIF && decompressFrames) {
                imgs.forEach(function(img) { initOne(img, parseGIF, decompressFrames); });
            }
        }).catch(function() {
            imgs.forEach(function(img) { img.style.display = ''; });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
