/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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

#ifndef WebKitDOMHTMLScriptElement_h
#define WebKitDOMHTMLScriptElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_SCRIPT_ELEMENT            (webkit_dom_html_script_element_get_type())
#define WEBKIT_DOM_HTML_SCRIPT_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_SCRIPT_ELEMENT, WebKitDOMHTMLScriptElement))
#define WEBKIT_DOM_HTML_SCRIPT_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_SCRIPT_ELEMENT, WebKitDOMHTMLScriptElementClass)
#define WEBKIT_DOM_IS_HTML_SCRIPT_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_SCRIPT_ELEMENT))
#define WEBKIT_DOM_IS_HTML_SCRIPT_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_SCRIPT_ELEMENT))
#define WEBKIT_DOM_HTML_SCRIPT_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_SCRIPT_ELEMENT, WebKitDOMHTMLScriptElementClass))

struct _WebKitDOMHTMLScriptElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLScriptElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_script_element_get_type(void);

/**
 * webkit_dom_html_script_element_get_text:
 * @self: A #WebKitDOMHTMLScriptElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_script_element_get_text(WebKitDOMHTMLScriptElement* self);

/**
 * webkit_dom_html_script_element_set_text:
 * @self: A #WebKitDOMHTMLScriptElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_script_element_set_text(WebKitDOMHTMLScriptElement* self, const gchar* value);

/**
 * webkit_dom_html_script_element_get_html_for:
 * @self: A #WebKitDOMHTMLScriptElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_script_element_get_html_for(WebKitDOMHTMLScriptElement* self);

/**
 * webkit_dom_html_script_element_set_html_for:
 * @self: A #WebKitDOMHTMLScriptElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_script_element_set_html_for(WebKitDOMHTMLScriptElement* self, const gchar* value);

/**
 * webkit_dom_html_script_element_get_event:
 * @self: A #WebKitDOMHTMLScriptElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_script_element_get_event(WebKitDOMHTMLScriptElement* self);

/**
 * webkit_dom_html_script_element_set_event:
 * @self: A #WebKitDOMHTMLScriptElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_script_element_set_event(WebKitDOMHTMLScriptElement* self, const gchar* value);

/**
 * webkit_dom_html_script_element_get_charset:
 * @self: A #WebKitDOMHTMLScriptElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_script_element_get_charset(WebKitDOMHTMLScriptElement* self);

/**
 * webkit_dom_html_script_element_set_charset:
 * @self: A #WebKitDOMHTMLScriptElement
 * @value: A #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_script_element_set_charset(WebKitDOMHTMLScriptElement* self, const gchar* value);

/**
 * webkit_dom_html_script_element_get_defer:
 * @self: A #WebKitDOMHTMLScriptElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_script_element_get_defer(WebKitDOMHTMLScriptElement* self);

/**
 * webkit_dom_html_script_element_set_defer:
 * @self: A #WebKitDOMHTMLScriptElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_script_element_set_defer(WebKitDOMHTMLScriptElement* self, gboolean value);

/**
 * webkit_dom_html_script_element_get_src:
 * @self: A #WebKitDOMHTMLScriptElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_script_element_get_src(WebKitDOMHTMLScriptElement* self);

/**
 * webkit_dom_html_script_element_set_src:
 * @self: A #WebKitDOMHTMLScriptElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_script_element_set_src(WebKitDOMHTMLScriptElement* self, const gchar* value);

/**
 * webkit_dom_html_script_element_get_type_attr:
 * @self: A #WebKitDOMHTMLScriptElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_script_element_get_type_attr(WebKitDOMHTMLScriptElement* self);

/**
 * webkit_dom_html_script_element_set_type_attr:
 * @self: A #WebKitDOMHTMLScriptElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_script_element_set_type_attr(WebKitDOMHTMLScriptElement* self, const gchar* value);

G_END_DECLS

#endif /* WebKitDOMHTMLScriptElement_h */
