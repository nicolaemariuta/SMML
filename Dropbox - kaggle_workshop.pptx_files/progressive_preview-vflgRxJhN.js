// Generated by CoffeeScript 1.7.1
define(["jquery", "modules/core/i18n", "modules/core/browser", "modules/core/uri"], function($j, _arg, Browser, URI) {
  var ProgressivePreview, _;
  _ = _arg._;
  return ProgressivePreview = {
    KEY_SCOPE: "progressivepreview",
    PAGE_CACHE_SIZE: 128,
    start_page: 1,
    page_count: 1,
    current_page: 0,
    menu_initialized: false,
    titles: [],
    saved_pages: [],
    is_pdf_js: true,
    is_progressive: true,
    is_private: true,
    is_full_doc_loaded: false,
    is_preview_available: false,
    print_pending: false,
    search_pending: false,
    wheel_time: 0,
    min_wheel_time: 150,
    search_pending_next: -1,
    text_content: [],
    sjid: 0,
    nav_prev: $j("#nav-prev"),
    nav_next: $j("#nav-next"),
    nav_screen: $j("#nav-screen"),
    nav_get: $j("#nav-get"),
    nav_print: $j("#nav-print"),
    nav_search: $j("#nav-search"),
    nav_search_text: $j("#nav-search-text"),
    nav_page_input: $j("#nav-page-input"),
    nav_num: $j("#nav-num"),
    init_file: function() {
      var preview_status, xhr, _complete_title_and_slide_loading, _doc_titles_callback, _metadata_fallback;
      this.is_private = window.location.pathname.indexOf('private_progressive_viewer') === 1;
      if (this.is_private) {
        this.sjid = window.parent.__PARENT_SCOPE__.FileViewer.file.sjid;
      }
      $j(window).on("message", (function(_this) {
        return function(e) {
          return _this._message_handler(e);
        };
      })(this));
      $j(window).on("keydown", (function(_this) {
        return function(e) {
          return _this._keydown_handler(e);
        };
      })(this));
      this.nav_prev.on("click", (function(_this) {
        return function() {
          return _this._prev_page();
        };
      })(this));
      this.nav_next.on("click", (function(_this) {
        return function() {
          return _this._next_page();
        };
      })(this));
      this.nav_screen.on("click", (function(_this) {
        return function() {
          return _this._toggle_full_screen();
        };
      })(this));
      this.nav_get.on("click", (function(_this) {
        return function() {
          return _this._toggle_full_doc();
        };
      })(this));
      this.nav_print.on("click", (function(_this) {
        return function() {
          return _this._print();
        };
      })(this));
      this.nav_search.on("click", function(e) {
        return e.stopPropagation();
      });
      this.nav_search_text.on("focus", (function(_this) {
        return function() {
          return $j("#nav").addClass("selected");
        };
      })(this));
      this.nav_search_text.on("blur", (function(_this) {
        return function() {
          return $j("#nav").removeClass("selected");
        };
      })(this));
      this.nav_page_input.on("focus", (function(_this) {
        return function() {
          return $j("#nav-page-input").addClass("selected");
        };
      })(this));
      this.nav_page_input.on("blur", (function(_this) {
        return function() {
          return $j("#nav-page-input").removeClass("selected");
        };
      })(this));
      this.nav_search.on("submit", (function(_this) {
        return function(e) {
          _this._search();
          return false;
        };
      })(this));
      this.nav_search.on("keydown", (function(_this) {
        return function(e) {
          if (e.keyCode === 27) {
            return _this._visible_frame().focus();
          }
        };
      })(this));
      $j("#nav").on("click", (function(_this) {
        return function() {
          if (!$j("#nav-page-input").is(":focus")) {
            return _this._visible_frame().focus();
          }
        };
      })(this));
      if (this.is_private) {
        preview_status = window.parent.__PARENT_SCOPE__.FileViewer.file.doc_preview_status;
      } else {
        preview_status = window.parent.__PARENT_SCOPE__.FilePreview.preview_status;
      }
      this.is_preview_available = preview_status === ProgressiveConfig.DOC_PREVIEW_AVAILABLE;
      if (this.is_preview_available) {
        this.nav_search.removeClass("hidden");
        this.nav_get.removeClass("hidden");
        this.nav_print.removeClass("hidden");
      }
      this._show_loading_icon();
      this._doc_page_count_callback = (function(_this) {
        return function(json_metadata) {
          return _this._metadata_setup(json_metadata['page_count']);
        };
      })(this);
      _doc_titles_callback = (function(_this) {
        return function(json_metadata) {
          var i, new_text, _i, _ref, _results;
          _this.titles = json_metadata['titles'];
          _this._metadata_setup(_this.titles.length);
          if (_this.titles.length === _this.page_count) {
            _results = [];
            for (i = _i = 0, _ref = _this.page_count; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
              new_text = $j("#title_link_" + i).text() + (_this.titles[i] ? ": " + _this.titles[i] : "");
              _results.push($j("#title_link_" + i).text(new_text));
            }
            return _results;
          }
        };
      })(this);
      _complete_title_and_slide_loading = (function(_this) {
        return function() {
          $j.ajax({
            url: String(_this._doc_titles_url()),
            xhrFields: {
              withCredentials: true
            }
          }).done(_doc_titles_callback);
          _this._generate_iframe(0);
          return _this._generate_iframe(1);
        };
      })(this);
      _metadata_fallback = (function(_this) {
        return function() {
          return _this._metadata_setup(200);
        };
      })(this);
      xhr = $j.ajax({
        url: String(this._doc_page_count_url()),
        xhrFields: {
          withCredentials: true
        }
      });
      if (this.is_private) {
        xhr.done(this._doc_page_count_callback);
        xhr.fail(_metadata_fallback);
        return _complete_title_and_slide_loading();
      } else {
        xhr.fail(this._fallback_to_no_progressive);
        return xhr.done(this._doc_page_count_callback).then(_complete_title_and_slide_loading);
      }
    },
    _get_new_url: function(endpoint, include_sjid, params) {

      /*global: ProgressiveConfig is defined in a Python String =( */
      var url;
      url = ProgressiveConfig.BLOCKSERVER;
      url += window.location.pathname.replace('progressive_viewer/', endpoint + '/');
      url += window.location.search;
      if (include_sjid && this.is_private) {
        params['sjid'] = this.sjid;
      }
      return URI.parse(url).updateQuery(params);
    },
    _doc_preview_url: function(page) {
      var url;
      url = this._get_new_url('doc_preview', true, {
        "start_page": page,
        "end_page": page + 1
      });
      if (this.is_pdf_js) {
        url = ProgressiveConfig.STATIC_PDFJS_SPP_VIEWER + '#file=' + encodeURIComponent(url);
      }
      return url;
    },
    _doc_get_url: function(download) {
      var static_url, url;
      if (this.is_private) {
        url = ProgressiveConfig.BLOCKSERVER;
        url += window.location.pathname.replace(/^\/[^/]*\//, '/get/');
        if (download) {
          url += window.location.search.replace('&get_preview=1', '');
        } else {
          url += window.location.search;
        }
        url = URI.parse(url).updateQuery({
          'sjid': this.sjid
        });
      } else {
        if (download) {
          url = URI.parse(Browser.get_href(window.parent)).updateQuery('dl', 1);
        } else {
          url = this._get_new_url('preview', false, {});
        }
      }
      if (!download && this.is_pdf_js) {
        static_url = ProgressiveConfig.STATIC_PDFJS_VIEWER;
        url = URI.parse(static_url).updateQuery('file', url).setFragment("progressive=1").toString();
      }
      return url;
    },
    _doc_page_count_url: function() {
      return this._get_new_url('doc_preview_page_count', true, {});
    },
    _doc_titles_url: function() {
      return this._get_new_url('doc_preview_titles', true, {});
    },
    _update_nav_metadata: function() {
      var i18nparams, nav_input_element, nav_num_element;
      if (this.current_page === 0) {
        this.nav_prev.addClass("disabled");
      } else {
        this.nav_prev.removeClass("disabled");
      }
      if (this.page_count > 1 && this.current_page + 1 === this.page_count) {
        this.nav_next.addClass("disabled");
      } else {
        this.nav_next.removeClass("disabled");
      }
      nav_num_element = this.nav_num;
      if (nav_num_element[0]) {
        i18nparams = {
          start_page: "" + (this.start_page + this.current_page),
          end_page: "" + (this.start_page + this.page_count - 1)
        };
        nav_num_element[0].textContent = _("of %(end_page)s").format(i18nparams);
      }
      nav_input_element = this.nav_page_input;
      if (nav_input_element[0]) {
        return nav_input_element[0].value = this.start_page + this.current_page;
      }
    },
    _metadata_setup: function(page_count) {
      var i, is_below_nav, li, link, title_link_prefix, ul, _i, _ref;
      this.nav_page_input[0].max = page_count;
      if (this.menu_initialized) {
        return;
      }
      this.menu_initialized = true;
      this.page_count = page_count;
      if (this.page_count === 1) {
        this._frame_for_page(1).remove;
      }
      is_below_nav = function(element) {
        var nav_bottom;
        nav_bottom = $j("#nav")[0].getBoundingClientRect().bottom;
        return element.getBoundingClientRect().bottom - element.clientHeight > nav_bottom;
      };
      ul = $j("<ul/>", {
        id: "titles-dropdown-menu"
      });
      ul.css("maxHeight", $j("#doc-preview")[0].clientHeight / 3 + "px");
      for (i = _i = 0, _ref = this.page_count; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
        title_link_prefix = 'title_link_';
        link = $j('<div id="' + title_link_prefix + i + '">Slide ' + (i + 1) + '</div>');
        link.on("click", (function(_this) {
          return function(e) {
            if (is_below_nav(e.target)) {
              _this._switch_page(parseInt(e.target.id.substr(title_link_prefix.length)));
              return e.stopPropagation();
            }
          };
        })(this));
        li = $j('<li/>');
        li.append(link);
        ul.append(li);
      }
      $j("#nav").append(ul);
      ul.on("mouseleave", function() {
        return this.style.display = "none";
      });
      this.nav_page_input.on("change", (function(_this) {
        return function(e) {
          return _this._switch_page(parseInt($j("#nav-page-input")[0].value) - _this.start_page);
        };
      })(this));
      this.nav_page_input.on("click", function(e) {
        return this.select();
      });
      this.nav_num.on("click", (function(_this) {
        return function() {
          var menu;
          menu = $j("#titles-dropdown-menu")[0];
          if (menu.style.display === "" || menu.style.display === "none") {
            menu.style.display = "inline";
          } else {
            menu.style.display = "none";
          }
          return _this._visible_frame().focus();
        };
      })(this));
      $j(window).on("resize", function() {
        return $j("#titles-dropdown-menu").css("maxHeight", $j("#doc-preview")[0].clientHeight / 3 + "px");
      });
      return this._update_nav_metadata();
    },
    _is_full_screen: function() {
      return document.webkitIsFullScreen || document.mozFullScreen || document.msFullScreenElement;
    },
    _toggle_full_screen: function() {
      var elem;
      if (this._is_full_screen()) {
        if (document.webkitExitFullscreen) {
          document.webkitExitFullscreen();
        } else if (document.mozCancelFullScreen) {
          document.mozCancelFullScreen();
        } else if (document.msExitFullscreen) {
          document.msExitFullscreen();
        }
      } else {
        elem = document.documentElement;
        if (elem.webkitRequestFullscreen) {
          elem.webkitRequestFullscreen();
        } else if (elem.mozRequestFullScreen) {
          elem.mozRequestFullScreen();
        } else if (elem.msRequestFullscreen) {
          elem.msRequestFullscreen();
        }
      }
      return this._visible_frame().focus();
    },
    _get_full_doc_frame: function() {
      var doc, url;
      doc = $j("#doc_full_preview");
      if (!doc.length) {
        url = this._doc_get_url(false);
        doc = $j("<iframe/>", {
          src: url,
          id: "doc_full_preview"
        });
        doc.addClass("invisible-page");
        $j("#doc-preview").append(doc);
      }
      return doc;
    },
    _toggle_full_doc: function() {
      var current_page, doc;
      this.is_progressive = !this.is_progressive;
      doc = this._get_full_doc_frame();
      current_page = this._frame_for_page(this.current_page);
      if (!this.is_progressive) {
        current_page.removeClass("visible-page");
        current_page.addClass("invisible-page");
        doc.removeClass("invisible-page");
        doc.addClass("visible-page");
        $j("#nav").addClass("full-doc-nav");
      } else {
        current_page.removeClass("invisible-page");
        current_page.addClass("visible-page");
        doc.removeClass("visible-page");
        doc.addClass("invisible-page");
        $j("#nav").removeClass("full-doc-nav");
      }
      return this._visible_frame().focus();
    },
    _print: function() {
      var doc;
      if (!this.is_preview_available) {
        return;
      }
      doc = this._get_full_doc_frame();
      if (this.is_full_doc_loaded) {
        return doc[0].contentWindow.postMessage('print', '*');
      } else {
        this._show_loading_icon();
        return this.print_pending = true;
      }
    },
    _post_search_message: function(iframe, force_search) {
      var query;
      query = this.nav_search_text[0].value.toLowerCase();
      return iframe.contentWindow.postMessage('{"action":"search",' + '"query":' + JSON.stringify(query) + ', ' + '"force":' + force_search + '}', '*');
    },
    _search: function() {
      this._post_search_message(this._visible_frame(), false);
      return this._get_full_doc_frame();
    },
    _find_next: function(query) {
      var p, page_num, _i, _ref, _ref1;
      for (p = _i = _ref = this.current_page + 1, _ref1 = this.current_page + this.page_count; _ref <= _ref1 ? _i <= _ref1 : _i >= _ref1; p = _ref <= _ref1 ? ++_i : --_i) {
        page_num = p % this.page_count;
        if (this.text_content[page_num].indexOf(query) !== -1) {
          return page_num;
        }
      }
      return null;
    },
    _search_next: function() {
      var p, page_num, query;
      if (!this.is_full_doc_loaded) {
        this._show_loading_icon();
        this.search_pending = true;
        return;
      }
      query = this.nav_search_text[0].value.toLowerCase();
      page_num = this._find_next(query);
      if (page_num === null) {
        return;
      }
      if (page_num === this.current_page) {
        this._visible_frame().contentWindow.postMessage('{"action":"clear"}', '*');
      } else {
        this._switch_page(page_num, true);
        p = this._find_next(query);
        if (p >= 0 && p < this.page_count && !this._frame_for_page(p).length) {
          this._generate_iframe(p);
        }
      }
      if (this.saved_pages.indexOf(page_num) > -1) {
        this._post_search_message(this._frame_for_page(page_num)[0], true);
        return this.nav_search_text[0].focus();
      } else {
        return this.search_pending_next = page_num;
      }
    },
    _keydown_handler: function(e) {
      var ctrl;
      ctrl = e.cntrlKey || e.metaKey;
      return this._validated_message_handler({
        'action': 'keydown',
        'keycode': e.keyCode,
        'ctrlKey': ctrl
      });
    },
    _message_handler: function(e) {
      var allowed_domains, message;
      allowed_domains = {
        "https://www.dropboxstatic.com": true,
        "https://block-dbdev.dev.corp.dropbox.com": true
      };
      if (!allowed_domains[e.originalEvent.origin]) {
        return;
      }
      try {
        message = JSON.parse(e.originalEvent.data);
        return this._validated_message_handler(message);
      } catch (_error) {}
    },
    _validated_message_handler: function(message) {
      var ctrlKey, delta, keycode, now, page;
      switch (message['action']) {
        case 'mousewheel':
          now = Date.now();
          if (now - this.wheel_time > this.min_wheel_time) {
            this.wheel_time = now;
            delta = message['delta'];
            if (delta > 0) {
              return this._prev_page();
            } else if (delta < 0) {
              return this._next_page();
            }
          }
          break;
        case 'keydown':
          keycode = message['keycode'];
          ctrlKey = message['ctrlKey'];
          if (ctrlKey) {
            switch (keycode) {
              case 70:
                if (this.is_preview_available && this.is_progressive) {
                  return this.nav_search_text[0].focus();
                }
                break;
              case 80:
                return this._print();
              case 83:
                return Browser.redirect(this._doc_get_url(true));
            }
          } else if (!this.nav_page_input.is(":focus")) {
            switch (keycode) {
              case 38:
              case 37:
              case 8:
              case 80:
                return this._prev_page();
              case 40:
              case 39:
              case 32:
              case 78:
                return this._next_page();
              case 13:
                if (this.nav_search_text[0].value.toLowerCase()) {
                  return this._search();
                } else {
                  return this._next_page();
                }
                break;
              case 27:
                if (this.is_private && !this._is_full_screen()) {
                  return window.parent.__PARENT_SCOPE__.FileViewer._hide();
                }
            }
          }
          break;
        case 'click':
          if (this._is_full_screen()) {
            return this._next_page();
          }
          break;
        case 'loaded':
          page = message['page'];
          if (page + 1 < this.page_count && (page + 1 > this.current_page) && (page - this.current_page < this.PAGE_CACHE_SIZE / 4) && (!this._frame_for_page(page + 1).length)) {
            this._generate_iframe(page + 1);
          }
          if (page === this.current_page) {
            this._set_visible(this._frame_for_page(page));
            if (this.search_pending_next === page) {
              this.search_pending_next = -1;
              this._post_search_message(this._frame_for_page(page)[0], true);
              this.nav_search_text[0].focus();
            }
          }
          return this._add_saved_page(page);
        case 'search':
          return this._search_next();
        case 'ready':
          this.is_full_doc_loaded = true;
          if (this.print_pending) {
            this.print_pending = false;
            this._hide_loading_icon();
            this._print();
          }
          this.text_content = message['data'];
          if (this.search_pending) {
            this._hide_loading_icon();
            return this._search_next();
          }
          break;
        case 'failed':
          if (message['status'] === 500 && message['page'] < 2) {
            if (this.is_private) {
              return window.parent.__PARENT_SCOPE__.FileViewer.render_file_failed();
            } else {
              return Browser.redirect(URI.parse(Browser.get_href(window.parent)).updateQuery('force_no_preview', 1), window.parent);
            }
          } else if (message['status'] === 403) {
            return this._fallback_to_no_progressive();
          }
      }
    },
    _fallback_to_no_progressive: function() {
      return Browser.redirect(URI.parse(Browser.get_href(window.parent)).updateQuery('force_no_progressive', 1), window.parent);
    },
    _set_visible: function(iframe) {
      var current_iframe;
      this._hide_loading_icon();
      current_iframe = this._visible_frame();
      if (!this.is_progressive || current_iframe === iframe[0]) {
        return;
      }
      iframe[0].contentWindow.postMessage('{"action":"clear"}', '*');
      iframe.removeClass("invisible-page");
      iframe.addClass("visible-page");
      if (!this.nav_page_input.is(":focus")) {
        iframe[0].focus();
      }
      $j(current_iframe).removeClass("visible-page");
      return $j(current_iframe).addClass("invisible-page");
    },
    _add_saved_page: function(page) {
      var index, remove_frame, remove_page;
      index = this.saved_pages.indexOf(page);
      if (index === -1) {
        if (this.saved_pages.unshift(page) > this.PAGE_CACHE_SIZE) {
          remove_page = this.saved_pages.pop();
          remove_frame = this._frame_for_page(remove_page);
          if (remove_frame[0].className.indexOf('invisible-page') > -1) {
            return remove_frame.remove();
          } else {
            return this._add_saved_page(remove_page);
          }
        }
      } else {
        return this.saved_pages = [page].concat(this.saved_pages.slice(0, index)).concat(this.saved_pages.slice(index + 1));
      }
    },
    _visible_frame: function() {
      return ($j(".visible-page"))[0];
    },
    _frame_for_page: function(page) {
      return $j("#doc_preview_page_" + page);
    },
    _generate_iframe: function(page) {
      var iframe, preview, url;
      preview = $j("#doc-preview");
      iframe = this._frame_for_page(page);
      if (!iframe.length) {
        url = this._doc_preview_url(page);
        iframe = $j("<iframe/>", {
          src: url,
          id: "doc_preview_page_" + page
        });
        iframe[0].style.minHeight = 0.75 * preview[0].offsetWidth;
        iframe.addClass("invisible-page");
        preview.append(iframe);
      }
      if (!this.is_pdf_js) {
        this._add_saved_page(page);
      }
      return iframe;
    },
    _show_loading_icon: function() {
      var icon;
      if ($j(".loading-icon").length === 0) {
        icon = $j("<div/>").addClass("loading-icon");
        return $j("#doc-preview").append(icon);
      }
    },
    _hide_loading_icon: function() {
      return $j(".loading-icon").remove();
    },
    _preload_iframes: function(page, search) {
      var offset, offsets_to_load, p, _i, _len, _results;
      if (search == null) {
        search = false;
      }
      offsets_to_load = [-1, 1];
      if (!search) {
        offsets_to_load.push(2, 3);
      }
      _results = [];
      for (_i = 0, _len = offsets_to_load.length; _i < _len; _i++) {
        offset = offsets_to_load[_i];
        p = page + offset;
        if (p >= 0 && p < this.page_count && !this._frame_for_page(p).length) {
          _results.push(this._generate_iframe(p));
        } else {
          _results.push(void 0);
        }
      }
      return _results;
    },
    _switch_page: function(page, search) {
      var pause_before_load;
      if (search == null) {
        search = false;
      }
      if (!this.is_progressive || page < 0 || page >= this.page_count || page === this.current_page) {
        return;
      }
      this.current_page = page;
      this._update_nav_metadata();
      $j("#titles-dropdown-menu")[0].style.display = "none";
      pause_before_load = 200;
      if (this.saved_pages.indexOf(page) > -1) {
        this._set_visible(this._frame_for_page(page));
        return setTimeout((function(_this) {
          return function() {
            if (page === _this.current_page) {
              return _this._preload_iframes(page, search);
            }
          };
        })(this), pause_before_load);
      } else {
        this._show_loading_icon();
        if (search) {
          this._generate_iframe(page);
          return this._preload_iframes(page, search);
        } else {
          return setTimeout((function(_this) {
            return function() {
              if (page === _this.current_page) {
                _this._generate_iframe(page);
                return _this._preload_iframes(page, search);
              }
            };
          })(this), pause_before_load);
        }
      }
    },
    _next_page: function() {
      return this._switch_page(this.current_page + 1);
    },
    _prev_page: function() {
      return this._switch_page(this.current_page - 1);
    }
  };
});

//# sourceMappingURL=progressive_preview.map
