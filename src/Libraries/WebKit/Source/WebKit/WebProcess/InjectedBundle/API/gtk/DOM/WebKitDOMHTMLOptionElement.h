/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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

#ifndef WebKitDOMHTMLOptionElement_h
#define WebKitDOMHTMLOptionElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_OPTION_ELEMENT            (webkit_dom_html_option_element_get_type())
#define WEBKIT_DOM_HTML_OPTION_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_OPTION_ELEMENT, WebKitDOMHTMLOptionElement))
#define WEBKIT_DOM_HTML_OPTION_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_OPTION_ELEMENT, WebKitDOMHTMLOptionElementClass)
#define WEBKIT_DOM_IS_HTML_OPTION_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_OPTION_ELEMENT))
#define WEBKIT_DOM_IS_HTML_OPTION_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_OPTION_ELEMENT))
#define WEBKIT_DOM_HTML_OPTION_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_OPTION_ELEMENT, WebKitDOMHTMLOptionElementClass))

struct _WebKitDOMHTMLOptionElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLOptionElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_option_element_get_type(void);

/**
 * webkit_dom_html_option_element_get_disabled:
 * @self: A #WebKitDOMHTMLOptionElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_option_element_get_disabled(WebKitDOMHTMLOptionElement* self);

/**
 * webkit_dom_html_option_element_set_disabled:
 * @self: A #WebKitDOMHTMLOptionElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_option_element_set_disabled(WebKitDOMHTMLOptionElement* self, gboolean value);

/**
 * webkit_dom_html_option_element_get_form:
 * @self: A #WebKitDOMHTMLOptionElement
 *
 * Returns: (transfer none): A #WebKitDOMHTMLFormElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLFormElement*
webkit_dom_html_option_element_get_form(WebKitDOMHTMLOptionElement* self);

/**
 * webkit_dom_html_option_element_get_label:
 * @self: A #WebKitDOMHTMLOptionElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_option_element_get_label(WebKitDOMHTMLOptionElement* self);

/**
 * webkit_dom_html_option_element_set_label:
 * @self: A #WebKitDOMHTMLOptionElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_option_element_set_label(WebKitDOMHTMLOptionElement* self, const gchar* value);

/**
 * webkit_dom_html_option_element_get_default_selected:
 * @self: A #WebKitDOMHTMLOptionElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_option_element_get_default_selected(WebKitDOMHTMLOptionElement* self);

/**
 * webkit_dom_html_option_element_set_default_selected:
 * @self: A #WebKitDOMHTMLOptionElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_option_element_set_default_selected(WebKitDOMHTMLOptionElement* self, gboolean value);

/**
 * webkit_dom_html_option_element_get_selected:
 * @self: A #WebKitDOMHTMLOptionElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_option_element_get_selected(WebKitDOMHTMLOptionElement* self);

/**
 * webkit_dom_html_option_element_set_selected:
 * @self: A #WebKitDOMHTMLOptionElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_option_element_set_selected(WebKitDOMHTMLOptionElement* self, gboolean value);

/**
 * webkit_dom_html_option_element_get_value:
 * @self: A #WebKitDOMHTMLOptionElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_option_element_get_value(WebKitDOMHTMLOptionElement* self);

/**
 * webkit_dom_html_option_element_set_value:
 * @self: A #WebKitDOMHTMLOptionElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_option_element_set_value(WebKitDOMHTMLOptionElement* self, const gchar* value);

/**
 * webkit_dom_html_option_element_get_text:
 * @self: A #WebKitDOMHTMLOptionElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_option_element_get_text(WebKitDOMHTMLOptionElement* self);

/**
 * webkit_dom_html_option_element_get_index:
 * @self: A #WebKitDOMHTMLOptionElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_option_element_get_index(WebKitDOMHTMLOptionElement* self);

G_END_DECLS

#endif /* WebKitDOMHTMLOptionElement_h */
