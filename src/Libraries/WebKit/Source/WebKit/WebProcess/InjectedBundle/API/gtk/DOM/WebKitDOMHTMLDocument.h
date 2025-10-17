/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#if !defined(__WEBKITDOM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <webkitdom/webkitdom.h> can be included directly."
#endif

#ifndef WebKitDOMHTMLDocument_h
#define WebKitDOMHTMLDocument_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMDocument.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_DOCUMENT            (webkit_dom_html_document_get_type())
#define WEBKIT_DOM_HTML_DOCUMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_DOCUMENT, WebKitDOMHTMLDocument))
#define WEBKIT_DOM_HTML_DOCUMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_DOCUMENT, WebKitDOMHTMLDocumentClass)
#define WEBKIT_DOM_IS_HTML_DOCUMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_DOCUMENT))
#define WEBKIT_DOM_IS_HTML_DOCUMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_DOCUMENT))
#define WEBKIT_DOM_HTML_DOCUMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_DOCUMENT, WebKitDOMHTMLDocumentClass))

struct _WebKitDOMHTMLDocument {
    WebKitDOMDocument parent_instance;
};

struct _WebKitDOMHTMLDocumentClass {
    WebKitDOMDocumentClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_document_get_type(void);

/**
 * webkit_dom_html_document_close:
 * @self: A #WebKitDOMHTMLDocument
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_close(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_clear:
 * @self: A #WebKitDOMHTMLDocument
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_clear(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_capture_events:
 * @self: A #WebKitDOMHTMLDocument
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_capture_events(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_release_events:
 * @self: A #WebKitDOMHTMLDocument
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_release_events(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_get_width:
 * @self: A #WebKitDOMHTMLDocument
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_document_get_width(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_get_height:
 * @self: A #WebKitDOMHTMLDocument
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_document_get_height(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_get_dir:
 * @self: A #WebKitDOMHTMLDocument
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_document_get_dir(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_set_dir:
 * @self: A #WebKitDOMHTMLDocument
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_set_dir(WebKitDOMHTMLDocument* self, const gchar* value);

/**
 * webkit_dom_html_document_get_bg_color:
 * @self: A #WebKitDOMHTMLDocument
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_document_get_bg_color(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_set_bg_color:
 * @self: A #WebKitDOMHTMLDocument
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_set_bg_color(WebKitDOMHTMLDocument* self, const gchar* value);

/**
 * webkit_dom_html_document_get_fg_color:
 * @self: A #WebKitDOMHTMLDocument
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_document_get_fg_color(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_set_fg_color:
 * @self: A #WebKitDOMHTMLDocument
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_set_fg_color(WebKitDOMHTMLDocument* self, const gchar* value);

/**
 * webkit_dom_html_document_get_alink_color:
 * @self: A #WebKitDOMHTMLDocument
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_document_get_alink_color(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_set_alink_color:
 * @self: A #WebKitDOMHTMLDocument
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_set_alink_color(WebKitDOMHTMLDocument* self, const gchar* value);

/**
 * webkit_dom_html_document_get_link_color:
 * @self: A #WebKitDOMHTMLDocument
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_document_get_link_color(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_set_link_color:
 * @self: A #WebKitDOMHTMLDocument
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_set_link_color(WebKitDOMHTMLDocument* self, const gchar* value);

/**
 * webkit_dom_html_document_get_vlink_color:
 * @self: A #WebKitDOMHTMLDocument
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_document_get_vlink_color(WebKitDOMHTMLDocument* self);

/**
 * webkit_dom_html_document_set_vlink_color:
 * @self: A #WebKitDOMHTMLDocument
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_document_set_vlink_color(WebKitDOMHTMLDocument* self, const gchar* value);

G_END_DECLS

#endif /* WebKitDOMHTMLDocument_h */
