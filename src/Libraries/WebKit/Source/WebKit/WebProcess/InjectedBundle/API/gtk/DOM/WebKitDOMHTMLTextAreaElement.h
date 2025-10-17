/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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

#ifndef WebKitDOMHTMLTextAreaElement_h
#define WebKitDOMHTMLTextAreaElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_TEXT_AREA_ELEMENT            (webkit_dom_html_text_area_element_get_type())
#define WEBKIT_DOM_HTML_TEXT_AREA_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_TEXT_AREA_ELEMENT, WebKitDOMHTMLTextAreaElement))
#define WEBKIT_DOM_HTML_TEXT_AREA_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_TEXT_AREA_ELEMENT, WebKitDOMHTMLTextAreaElementClass)
#define WEBKIT_DOM_IS_HTML_TEXT_AREA_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_TEXT_AREA_ELEMENT))
#define WEBKIT_DOM_IS_HTML_TEXT_AREA_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_TEXT_AREA_ELEMENT))
#define WEBKIT_DOM_HTML_TEXT_AREA_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_TEXT_AREA_ELEMENT, WebKitDOMHTMLTextAreaElementClass))

struct _WebKitDOMHTMLTextAreaElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLTextAreaElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_text_area_element_get_type(void);

/**
 * webkit_dom_html_text_area_element_select:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_select(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_selection_range:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @start: A #glong
 * @end: A #glong
 * @direction: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_selection_range(WebKitDOMHTMLTextAreaElement* self, glong start, glong end, const gchar* direction);

/**
 * webkit_dom_html_text_area_element_get_autofocus:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_text_area_element_get_autofocus(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_autofocus:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_autofocus(WebKitDOMHTMLTextAreaElement* self, gboolean value);

/**
 * webkit_dom_html_text_area_element_get_disabled:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_text_area_element_get_disabled(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_disabled:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_disabled(WebKitDOMHTMLTextAreaElement* self, gboolean value);

/**
 * webkit_dom_html_text_area_element_get_form:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: (transfer none): A #WebKitDOMHTMLFormElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLFormElement*
webkit_dom_html_text_area_element_get_form(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_get_name:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_text_area_element_get_name(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_name:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_name(WebKitDOMHTMLTextAreaElement* self, const gchar* value);

/**
 * webkit_dom_html_text_area_element_get_read_only:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_text_area_element_get_read_only(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_read_only:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_read_only(WebKitDOMHTMLTextAreaElement* self, gboolean value);

/**
 * webkit_dom_html_text_area_element_get_rows:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_text_area_element_get_rows(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_rows:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_rows(WebKitDOMHTMLTextAreaElement* self, glong value);

/**
 * webkit_dom_html_text_area_element_get_cols:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_text_area_element_get_cols(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_cols:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_cols(WebKitDOMHTMLTextAreaElement* self, glong value);

/**
 * webkit_dom_html_text_area_element_get_area_type:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_text_area_element_get_area_type(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_get_default_value:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_text_area_element_get_default_value(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_default_value:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_default_value(WebKitDOMHTMLTextAreaElement* self, const gchar* value);

/**
 * webkit_dom_html_text_area_element_get_value:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_text_area_element_get_value(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_value:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_value(WebKitDOMHTMLTextAreaElement* self, const gchar* value);

/**
 * webkit_dom_html_text_area_element_get_will_validate:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_text_area_element_get_will_validate(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_get_selection_start:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_text_area_element_get_selection_start(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_selection_start:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_selection_start(WebKitDOMHTMLTextAreaElement* self, glong value);

/**
 * webkit_dom_html_text_area_element_get_selection_end:
 * @self: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_text_area_element_get_selection_end(WebKitDOMHTMLTextAreaElement* self);

/**
 * webkit_dom_html_text_area_element_set_selection_end:
 * @self: A #WebKitDOMHTMLTextAreaElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_text_area_element_set_selection_end(WebKitDOMHTMLTextAreaElement* self, glong value);

/**
 * webkit_dom_html_text_area_element_is_edited:
 * @input: A #WebKitDOMHTMLTextAreaElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean webkit_dom_html_text_area_element_is_edited(WebKitDOMHTMLTextAreaElement* input);

G_END_DECLS

#endif /* WebKitDOMHTMLTextAreaElement_h */
