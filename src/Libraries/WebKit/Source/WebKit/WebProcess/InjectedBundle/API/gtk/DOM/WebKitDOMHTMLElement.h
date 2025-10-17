/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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

#ifndef WebKitDOMHTMLElement_h
#define WebKitDOMHTMLElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_ELEMENT            (webkit_dom_html_element_get_type())
#define WEBKIT_DOM_HTML_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_ELEMENT, WebKitDOMHTMLElement))
#define WEBKIT_DOM_HTML_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_ELEMENT, WebKitDOMHTMLElementClass)
#define WEBKIT_DOM_IS_HTML_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_ELEMENT))
#define WEBKIT_DOM_IS_HTML_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_ELEMENT))
#define WEBKIT_DOM_HTML_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_ELEMENT, WebKitDOMHTMLElementClass))

struct _WebKitDOMHTMLElement {
    WebKitDOMElement parent_instance;
};

struct _WebKitDOMHTMLElementClass {
    WebKitDOMElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_element_get_type(void);

/**
 * webkit_dom_html_element_click:
 * @self: A #WebKitDOMHTMLElement
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_click(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_get_title:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_element_get_title(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_title:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_title(WebKitDOMHTMLElement* self, const gchar* value);

/**
 * webkit_dom_html_element_get_lang:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_element_get_lang(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_lang:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_lang(WebKitDOMHTMLElement* self, const gchar* value);

/**
 * webkit_dom_html_element_get_dir:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_element_get_dir(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_dir:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_dir(WebKitDOMHTMLElement* self, const gchar* value);

/**
 * webkit_dom_html_element_get_tab_index:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_element_get_tab_index(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_tab_index:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_tab_index(WebKitDOMHTMLElement* self, glong value);

/**
 * webkit_dom_html_element_get_access_key:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_element_get_access_key(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_access_key:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_access_key(WebKitDOMHTMLElement* self, const gchar* value);

/**
 * webkit_dom_html_element_get_inner_text:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_element_get_inner_text(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_inner_text:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_inner_text(WebKitDOMHTMLElement* self, const gchar* value, GError** error);

/**
 * webkit_dom_html_element_get_outer_text:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_element_get_outer_text(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_outer_text:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_outer_text(WebKitDOMHTMLElement* self, const gchar* value, GError** error);

/**
 * webkit_dom_html_element_get_content_editable:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_element_get_content_editable(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_content_editable:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_content_editable(WebKitDOMHTMLElement* self, const gchar* value, GError** error);

/**
 * webkit_dom_html_element_get_is_content_editable:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_element_get_is_content_editable(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_get_translate:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_element_get_translate(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_translate:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_translate(WebKitDOMHTMLElement* self, gboolean value);

/**
 * webkit_dom_html_element_get_draggable:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_element_get_draggable(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_draggable:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_draggable(WebKitDOMHTMLElement* self, gboolean value);

/**
 * webkit_dom_html_element_get_webkitdropzone:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_element_get_webkitdropzone(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_webkitdropzone:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_webkitdropzone(WebKitDOMHTMLElement* self, const gchar* value);

/**
 * webkit_dom_html_element_get_hidden:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_element_get_hidden(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_hidden:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_hidden(WebKitDOMHTMLElement* self, gboolean value);

/**
 * webkit_dom_html_element_get_spellcheck:
 * @self: A #WebKitDOMHTMLElement
 *
 * Returns: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_element_get_spellcheck(WebKitDOMHTMLElement* self);

/**
 * webkit_dom_html_element_set_spellcheck:
 * @self: A #WebKitDOMHTMLElement
 * @value: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_element_set_spellcheck(WebKitDOMHTMLElement* self, gboolean value);

G_END_DECLS

#endif /* WebKitDOMHTMLElement_h */
