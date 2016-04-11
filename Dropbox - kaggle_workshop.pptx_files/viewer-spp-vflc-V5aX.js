var PDFPage;
var PDFPageNumber;
var lastQuery;

// Resize the page element to make the PDF preview as large as possible
var resize = function () {

  var currentWidth = parseFloat(PDFPage.style.width);
  var currentHeight = parseFloat(PDFPage.style.height);

  // Keep the same aspect ratio for the PDF
  var aspectRatio = currentWidth / currentHeight;
  // Leave some space on either side, to prevent creating a scrollbar
  var bufferSpace = 10;
  var newWidth = Math.min(window.innerWidth, window.innerHeight * aspectRatio) - bufferSpace;
  var scale = newWidth / currentWidth;

  if (scale == 1) {
    return;
  }

  PDFPage.style.width = newWidth + "px";
  PDFPage.style.height = (newWidth / aspectRatio) + "px";

  // Update left, top, transform style elements to scale text layer divs
  var textLayer = document.getElementsByClassName("textLayer")[0];
  var textDivs = textLayer.children;

  for(var i = 0; i < textDivs.length; i++) {
    var textDiv = textDivs[i];

    textDiv.style.left = scale * parseFloat(textDiv.style.left);
    textDiv.style.top = scale * parseFloat(textDiv.style.top);

    // Text layer uses style.transform in Chrome / FF, style.webkitTransform in Safari
    var transform = textDiv.style.transform || textDiv.style.webkitTransform;

    // Change values in scale transform
    var regex = /scale\(([0-9.]+), ([0-9.]+)\)/
    var matches = regex.exec(transform);
    var newXTransform = parseFloat(matches[1]) * scale;
    var newYTransform = parseFloat(matches[2]) * scale;

    transform = "scale(" + newXTransform + ", " + newYTransform + ")";
    if (textDiv.style.transform) {
      textDiv.style.transform = transform;
    } else if (textDiv.style.webkitTransform) {
      textDiv.style.webkitTransform = transform;
    }
  }
  var xcanvas = null;
  var pdfPageChildren = PDFPage.children;
  for (var i = pdfPageChildren.length - 1; i >= 0; --i) {
    var nn = pdfPageChildren[i].nodeName;
    if (nn && nn.toLowerCase() == "canvas") {
      xcanvas = pdfPageChildren[i];
    }
  }
  var viewportWidth = parseInt(xcanvas.getAttribute("data-viewport-width"));
  var viewportHeight = parseInt(xcanvas.getAttribute("data-viewport-height"));
  var scaleX = PDFPage.clientWidth / viewportWidth;
  var xviewport = new PDFJS.PageViewport([0,0,viewportHeight, viewportWidth],
                                         scaleX, 90, 0, 0)
  var annotationLayers = PDFPage.getElementsByClassName("annotationLayer");
  var viewport = xviewport.clone({ dontFlip: true });
  var transformStr = 'matrix(' + viewport.transform.join(',') + ')';
  for (var i=annotationLayers.length - 1; i >= 0; --i) {
    var links = annotationLayers[i].children;
    for (var j=links.length -1; j >=0; --j) {
      CustomStyle.setProp('transform', links[j], transformStr);
    }
  }
};

window.onload = function () {

  var scale = 1; //This is basically the "zoom" factor for the PDF.
  PDFJS.workerSrc = pdfWorkerSrc;

  function loadPdf(pdfPath) {
    var pdf = PDFJS.getDocument(pdfPath);
    return pdf.then(renderPdf, handleFailedSinglePagePreview);
  }

  function handleFailedSinglePagePreview(err) {
    // Error message is of the form:
    // "Unexpected server response (###) while retrieving PDF ...
    window.parent.postMessage('{"action":"failed",' +
                              ' "status":' + err.substr(28, 3) + ',' +
                              ' "page":' + PDFPageNumber + '}', '*');
    document.body.className = 'viewer-failed';
  }

  function renderPdf(pdf) {
    return pdf.getPage(1).then(renderPage);
  }
  function setupAnnotations(pageDiv, pdfPage, viewport) {

    function bindLink(link, dest) {
      link.href = PDFView.getDestinationHash(dest);
      link.onclick = function pageViewSetupLinksOnclick() {
        if (dest) {
          PDFView.navigateTo(dest);
        }
        return false;
      };
      if (dest) {
        link.className = 'internalLink';
      }
    }

    function bindNamedAction(link, action) {
      link.href = PDFView.getAnchorUrl('');
      link.onclick = function pageViewSetupNamedActionOnClick() {
        // See PDF reference, table 8.45 - Named action
        switch (action) {
          case 'GoToPage':
            document.getElementById('pageNumber').focus();
            break;

          case 'GoBack':
            PDFHistory.back();
            break;

          case 'GoForward':
            PDFHistory.forward();
            break;

          case 'Find':
            if (!PDFView.supportsIntegratedFind) {
              PDFFindBar.toggle();
            }
            break;

          case 'NextPage':
            PDFView.page++;
            break;

          case 'PrevPage':
            PDFView.page--;
            break;

          case 'LastPage':
            PDFView.page = PDFView.pages.length;
            break;

          case 'FirstPage':
            PDFView.page = 1;
            break;

          default:
            break; // No action according to spec
        }
        return false;
      };
      link.className = 'internalLink';
    }
    pdfPage.getAnnotations().then(function(annotationsData) {
      viewport = viewport.clone({ dontFlip: true });
      var transform = viewport.transform;
      var transformStr = 'matrix(' + transform.join(',') + ')';
      var data, element, i, ii;

      for (i = 0, ii = annotationsData.length; i < ii; i++) {
        data = annotationsData[i];
        if (!data || !data.hasHtml) {
          continue;
        }

        element = PDFJS.AnnotationUtils.getHtmlElement(data,
                                                       pdfPage.commonObjs);
        element.setAttribute('data-annotation-id', data.id);
        //var mozL10n = document.mozL10n || document.webL10n;
        //mozL10n.translate(element);

        var rect = data.rect;
        var view = pdfPage.view;
        rect = PDFJS.Util.normalizeRect([
          rect[0],
          view[3] - rect[1] + view[1],
          rect[2],
          view[3] - rect[3] + view[1]
        ]);
        element.style.left = rect[0] + 'px';
        element.style.top = rect[1] + 'px';
        element.style.position = 'absolute';

        CustomStyle.setProp('transform', element, transformStr);
        var transformOriginStr = -rect[0] + 'px ' + -rect[1] + 'px';
        CustomStyle.setProp('transformOrigin', element, transformOriginStr);

        if (data.subtype === 'Link' && !data.url) {
          var link = element.getElementsByTagName('a')[0];
          if (link) {
            if (data.action) {
              bindNamedAction(link, data.action);
            } else {
              bindLink(link, ('dest' in data) ? data.dest : null);
            }
          }
        }
        var annotationLayerDiv = pdfPage.selfAnnotationLayer;
        if (!pdfPage.selfAnnotationLayer) {
          annotationLayerDiv = document.createElement('div');
          annotationLayerDiv.className = 'annotationLayer';
          pageDiv.appendChild(annotationLayerDiv);
          pdfPage.selfAnnotationLayer = annotationLayerDiv;
        }

        annotationLayerDiv.appendChild(element);
      }
    });
  }
  function renderPage(page) {
    var viewport = page.getViewport(scale);

    // Create and append the 'pdf-page' div to the pdf container.
    PDFPage = document.createElement('div');
    PDFPage.className = 'page';
    var pdfContainer = document.getElementById('pdfContainer');
    pdfContainer.appendChild(PDFPage);

    // Set the canvas height and width to the height and width of the viewport.
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');

    // The following few lines of code set up scaling on the context, if we are
    // on a HiDPI display.
    var outputScale = getOutputScale(context);
    canvas.setAttribute("data-viewport-width", viewport.width);
    canvas.setAttribute("data-viewport-height", viewport.height);
    targetHeight = (document.body.clientHeight - 4) * outputScale.sy;
    targetWidth = targetHeight * viewport.width / viewport.height;
    context._scaleX = targetWidth / viewport.width;
    context._scaleY = targetHeight / viewport.height;
    outputScale.scaled = true;
    if (context._scaleX < 1 || context._scaleX < 1) {
        // make sure it is at least a minimum rez
        context._scaleX = 1;
        context._scaleY = 1;
        targetWidth = viewport.width;
        targetHeight = viewport.height;
        outputScale.scaled = false;
    } // but in other cases clarify the display for a 1:1 ratio
    canvas.width = targetWidth | 0;
    canvas.height = targetHeight | 0;
    if (outputScale.scaled) {
      context.scale(context._scaleX, context._scaleY);
    }

    // The page, canvas and text layer elements will have the same size.
    canvas.style.width = '100%';
    canvas.style.height = '100%';

    PDFPage.style.width = Math.floor(viewport.width) + 'px';
    PDFPage.style.height = Math.floor(viewport.height) + 'px';
    PDFPage.appendChild(canvas);

    var textLayerDiv = document.createElement('div');
    textLayerDiv.className = 'textLayer';
    textLayerDiv.style.width = canvas.style.width;
    textLayerDiv.style.height = canvas.style.height;
    PDFPage.appendChild(textLayerDiv);

    // Painting the canvas...
    var renderContext = {
      canvasContext: context,
      viewport: viewport
    };
    var renderTask = page.render(renderContext);

    // ... and at the same time, getting the text and creating the text layer.
    var textLayerPromise = page.getTextContent().then(function (textContent) {
      var textLayerBuilder = new TextLayerBuilder({
        textLayerDiv: textLayerDiv,
        viewport: viewport,
        pageIndex: 0,
      });
      textLayerBuilder.setTextContent(textContent);
      var scaleX = PDFPage.clientWidth / canvas.width;
      var yviewport = page.getViewport(scaleX);
      // this sets up the annotations for the scale computed by the ratio of canvas to client width
      setupAnnotations(PDFPage, page, yviewport);
    });
    // We might be interested when rendering complete and text layer is built.
    return Promise.all([renderTask.promise, textLayerPromise]);
  }

  // Read file from query parameter
  pdfPath = window.location.href.match(/#file=([^&]+)/)[1];
  pdfPath = decodeURIComponent(pdfPath);
  PDFPageNumber = pdfPath.match('start_page=([0-9]+)')[1];

  loadPdf(pdfPath).then(function() {
    // Resize page once, and set up correct behavior on future resizes
    resize();
    window.onresize = resize;
    // Send a message to the parent telling it that page has loaded
    window.parent.postMessage('{"action":"loaded", "page":' + PDFPageNumber + '}', '*');
  });

};

// Send keypresses to parent
window.onkeydown = function (e) {
  window.parent.postMessage('{"action":"keydown",' +
                             '"keycode":' + e.keyCode + ',' +
                             '"ctrlKey":' + (e.ctrlKey || e.metaKey) + '}', '*');
  return false;
};

// Send keypresses to parent
var mouseWheelHandler = function (e) {
  var delta = e.wheelDelta || 0;
  // wheel scaling factor from https://developer.mozilla.org/en-US/docs/Web/Events/mousewheel
  if (e.deltaY) {
    delta = -e.deltaY;
  }
  if (e.detail) {
    delta = -e.detail;
  }
  if (Math.abs(delta) >= 40) {
    delta /= 40.0;
  }
  if (Math.abs(delta) <= 1./40) {
    delta *= 40.0;
  }
  window.parent.postMessage('{"action":"mousewheel",' +
                            '"delta":' + delta + '}', '*');
  return false;
};

window.onmousewheel = mouseWheelHandler;
window.onwheel = mouseWheelHandler;
// Send clicks to parent
window.onclick = function (e) {
  window.parent.postMessage('{"action":"click"}', '*');
};

// Handle messages from smart frame for search
window.onmessage = function (e) {
  var allowed_domains = {
    "https://www.dropbox.com": true,
    "https://meta-dbdev.dev.corp.dropbox.com": true
  };
  if (allowed_domains[e.origin]) {
    try {
      var message = JSON.parse(e.data);
      switch (message['action']) {
        case 'clear':
          // Remove selected text to search from beginning
          // Called when making a slide visible
          window.getSelection().removeAllRanges();
          break;
        case 'search':
          var query = message['query'];
          var forceSearch = message['force'];
          var searchFailed = false;
          // If this message is received, ProgressivePreview thinks
          // the query is here. If force_search is set, allow one search.
          // Otherwise, send a message to parent informing it to move on.
          if (!window.find(query) && (!forceSearch || lastQuery === query)) {
            // Reached the last occurence of query on page.
            window.parent.postMessage('{"action":"search"}', '*');
            var searchFailed = true;
          }
          lastQuery = (searchFailed) ? null : query;
          break;
      }
    } catch (error) {
      // Do nothing
    }

  }
};

