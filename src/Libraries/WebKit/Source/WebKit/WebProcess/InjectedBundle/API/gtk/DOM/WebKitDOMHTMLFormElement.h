/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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

#ifndef WebKitDOMHTMLFormElement_h
#define WebKitDOMHTMLFormElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_FORM_ELEMENT            (webkit_dom_html_form_element_get_type())
#define WEBKIT_DOM_HTML_FORM_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_FORM_ELEMENT, WebKitDOMHTMLFormElement))
#define WEBKIT_DOM_HTML_FORM_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_FORM_ELEMENT, WebKitDOMHTMLFormElementClass)
#define WEBKIT_DOM_IS_HTML_FORM_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_FORM_ELEMENT))
#define WEBKIT_DOM_IS_HTML_FORM_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_FORM_ELEMENT))
#define WEBKIT_DOM_HTML_FORM_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_FORM_ELEMENT, WebKitDOMHTMLFormElementClass))

struct _WebKitDOMHTMLFormElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLFormElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_form_element_get_type(void);

/**
 * webkit_dom_html_form_element_submit:
 * @self: A #WebKitDOMHTMLFormElement
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_submit(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_reset:
 * @self: A #WebKitDOMHTMLFormElement
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_reset(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_get_accept_charset:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_form_element_get_accept_charset(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_set_accept_charset:
 * @self: A #WebKitDOMHTMLFormElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_set_accept_charset(WebKitDOMHTMLFormElement* self, const gchar* value);

/**
 * webkit_dom_html_form_element_get_action:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_form_element_get_action(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_set_action:
 * @self: A #WebKitDOMHTMLFormElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_set_action(WebKitDOMHTMLFormElement* self, const gchar* value);

/**
 * webkit_dom_html_form_element_get_enctype:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_form_element_get_enctype(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_set_enctype:
 * @self: A #WebKitDOMHTMLFormElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_set_enctype(WebKitDOMHTMLFormElement* self, const gchar* value);

/**
 * webkit_dom_html_form_element_get_encoding:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_form_element_get_encoding(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_set_encoding:
 * @self: A #WebKitDOMHTMLFormElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_set_encoding(WebKitDOMHTMLFormElement* self, const gchar* value);

/**
 * webkit_dom_html_form_element_get_method:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_form_element_get_method(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_set_method:
 * @self: A #WebKitDOMHTMLFormElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_set_method(WebKitDOMHTMLFormElement* self, const gchar* value);

/**
 * webkit_dom_html_form_element_get_name:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_form_element_get_name(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_set_name:
 * @self: A #WebKitDOMHTMLFormElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_set_name(WebKitDOMHTMLFormElement* self, const gchar* value);

/**
 * webkit_dom_html_form_element_get_target:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_form_element_get_target(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_set_target:
 * @self: A #WebKitDOMHTMLFormElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_form_element_set_target(WebKitDOMHTMLFormElement* self, const gchar* value);

/**
 * webkit_dom_html_form_element_get_elements:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: (transfer full): A #WebKitDOMHTMLCollection
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLCollection*
webkit_dom_html_form_element_get_elements(WebKitDOMHTMLFormElement* self);

/**
 * webkit_dom_html_form_element_get_length:
 * @self: A #WebKitDOMHTMLFormElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_form_element_get_length(WebKitDOMHTMLFormElement* self);

G_END_DECLS

#endif /* WebKitDOMHTMLFormElement_h */
