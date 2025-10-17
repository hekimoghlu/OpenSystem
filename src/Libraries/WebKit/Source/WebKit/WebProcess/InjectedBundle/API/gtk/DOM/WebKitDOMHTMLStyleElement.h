/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

#ifndef WebKitDOMHTMLStyleElement_h
#define WebKitDOMHTMLStyleElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_STYLE_ELEMENT            (webkit_dom_html_style_element_get_type())
#define WEBKIT_DOM_HTML_STYLE_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_STYLE_ELEMENT, WebKitDOMHTMLStyleElement))
#define WEBKIT_DOM_HTML_STYLE_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_STYLE_ELEMENT, WebKitDOMHTMLStyleElementClass)
#define WEBKIT_DOM_IS_HTML_STYLE_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_STYLE_ELEMENT))
#define WEBKIT_DOM_IS_HTML_STYLE_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_STYLE_ELEMENT))
#define WEBKIT_DOM_HTML_STYLE_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_STYLE_ELEMENT, WebKitDOMHTMLStyleElementClass))

struct _WebKitDOMHTMLStyleElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLStyleElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_style_element_get_type(void);

/**
 * webkit_dom_html_style_element_get_disabled:
 * @self: A #WebKitDOMHTMLStyleElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_style_element_get_disabled(WebKitDOMHTMLStyleElement* self);

/**
 * webkit_dom_html_style_element_set_disabled:
 * @self: A #WebKitDOMHTMLStyleElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_style_element_set_disabled(WebKitDOMHTMLStyleElement* self, gboolean value);

/**
 * webkit_dom_html_style_element_get_media:
 * @self: A #WebKitDOMHTMLStyleElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_style_element_get_media(WebKitDOMHTMLStyleElement* self);

/**
 * webkit_dom_html_style_element_set_media:
 * @self: A #WebKitDOMHTMLStyleElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_style_element_set_media(WebKitDOMHTMLStyleElement* self, const gchar* value);

/**
 * webkit_dom_html_style_element_get_type_attr:
 * @self: A #WebKitDOMHTMLStyleElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_style_element_get_type_attr(WebKitDOMHTMLStyleElement* self);

/**
 * webkit_dom_html_style_element_set_type_attr:
 * @self: A #WebKitDOMHTMLStyleElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_style_element_set_type_attr(WebKitDOMHTMLStyleElement* self, const gchar* value);

/**
 * webkit_dom_html_style_element_get_sheet:
 * @self: A #WebKitDOMHTMLStyleElement
 *
 * Returns: (transfer full): A #WebKitDOMStyleSheet
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMStyleSheet*
webkit_dom_html_style_element_get_sheet(WebKitDOMHTMLStyleElement* self);

G_END_DECLS

#endif /* WebKitDOMHTMLStyleElement_h */
