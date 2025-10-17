/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#ifndef WebKitDOMHTMLAppletElement_h
#define WebKitDOMHTMLAppletElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_APPLET_ELEMENT            (webkit_dom_html_applet_element_get_type())
#define WEBKIT_DOM_HTML_APPLET_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_APPLET_ELEMENT, WebKitDOMHTMLAppletElement))
#define WEBKIT_DOM_HTML_APPLET_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_APPLET_ELEMENT, WebKitDOMHTMLAppletElementClass)
#define WEBKIT_DOM_IS_HTML_APPLET_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_APPLET_ELEMENT))
#define WEBKIT_DOM_IS_HTML_APPLET_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_APPLET_ELEMENT))
#define WEBKIT_DOM_HTML_APPLET_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_APPLET_ELEMENT, WebKitDOMHTMLAppletElementClass))

struct _WebKitDOMHTMLAppletElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLAppletElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_applet_element_get_type(void);

/**
 * webkit_dom_html_applet_element_get_align:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_align(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_align:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_align(WebKitDOMHTMLAppletElement* self, const gchar* value);

/**
 * webkit_dom_html_applet_element_get_alt:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_alt(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_alt:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_alt(WebKitDOMHTMLAppletElement* self, const gchar* value);

/**
 * webkit_dom_html_applet_element_get_archive:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_archive(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_archive:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_archive(WebKitDOMHTMLAppletElement* self, const gchar* value);

/**
 * webkit_dom_html_applet_element_get_code:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_code(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_code:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_code(WebKitDOMHTMLAppletElement* self, const gchar* value);

/**
 * webkit_dom_html_applet_element_get_code_base:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_code_base(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_code_base:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_code_base(WebKitDOMHTMLAppletElement* self, const gchar* value);

/**
 * webkit_dom_html_applet_element_get_height:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_height(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_height:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_height(WebKitDOMHTMLAppletElement* self, const gchar* value);

/**
 * webkit_dom_html_applet_element_get_hspace:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_applet_element_get_hspace(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_hspace:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_hspace(WebKitDOMHTMLAppletElement* self, glong value);

/**
 * webkit_dom_html_applet_element_get_name:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_name(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_name:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_name(WebKitDOMHTMLAppletElement* self, const gchar* value);

/**
 * webkit_dom_html_applet_element_get_object:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_object(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_object:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_object(WebKitDOMHTMLAppletElement* self, const gchar* value);

/**
 * webkit_dom_html_applet_element_get_vspace:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_applet_element_get_vspace(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_vspace:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_vspace(WebKitDOMHTMLAppletElement* self, glong value);

/**
 * webkit_dom_html_applet_element_get_width:
 * @self: A #WebKitDOMHTMLAppletElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_applet_element_get_width(WebKitDOMHTMLAppletElement* self);

/**
 * webkit_dom_html_applet_element_set_width:
 * @self: A #WebKitDOMHTMLAppletElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_applet_element_set_width(WebKitDOMHTMLAppletElement* self, const gchar* value);

G_END_DECLS

#endif /* WebKitDOMHTMLAppletElement_h */
