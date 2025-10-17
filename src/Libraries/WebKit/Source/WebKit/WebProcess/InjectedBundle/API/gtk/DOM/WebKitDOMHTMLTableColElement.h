/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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

#ifndef WebKitDOMHTMLTableColElement_h
#define WebKitDOMHTMLTableColElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_TABLE_COL_ELEMENT            (webkit_dom_html_table_col_element_get_type())
#define WEBKIT_DOM_HTML_TABLE_COL_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_TABLE_COL_ELEMENT, WebKitDOMHTMLTableColElement))
#define WEBKIT_DOM_HTML_TABLE_COL_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_TABLE_COL_ELEMENT, WebKitDOMHTMLTableColElementClass)
#define WEBKIT_DOM_IS_HTML_TABLE_COL_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_TABLE_COL_ELEMENT))
#define WEBKIT_DOM_IS_HTML_TABLE_COL_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_TABLE_COL_ELEMENT))
#define WEBKIT_DOM_HTML_TABLE_COL_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_TABLE_COL_ELEMENT, WebKitDOMHTMLTableColElementClass))

struct _WebKitDOMHTMLTableColElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLTableColElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_table_col_element_get_type(void);

/**
 * webkit_dom_html_table_col_element_get_align:
 * @self: A #WebKitDOMHTMLTableColElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_col_element_get_align(WebKitDOMHTMLTableColElement* self);

/**
 * webkit_dom_html_table_col_element_set_align:
 * @self: A #WebKitDOMHTMLTableColElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_col_element_set_align(WebKitDOMHTMLTableColElement* self, const gchar* value);

/**
 * webkit_dom_html_table_col_element_get_ch:
 * @self: A #WebKitDOMHTMLTableColElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_col_element_get_ch(WebKitDOMHTMLTableColElement* self);

/**
 * webkit_dom_html_table_col_element_set_ch:
 * @self: A #WebKitDOMHTMLTableColElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_col_element_set_ch(WebKitDOMHTMLTableColElement* self, const gchar* value);

/**
 * webkit_dom_html_table_col_element_get_ch_off:
 * @self: A #WebKitDOMHTMLTableColElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_col_element_get_ch_off(WebKitDOMHTMLTableColElement* self);

/**
 * webkit_dom_html_table_col_element_set_ch_off:
 * @self: A #WebKitDOMHTMLTableColElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_col_element_set_ch_off(WebKitDOMHTMLTableColElement* self, const gchar* value);

/**
 * webkit_dom_html_table_col_element_get_span:
 * @self: A #WebKitDOMHTMLTableColElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_table_col_element_get_span(WebKitDOMHTMLTableColElement* self);

/**
 * webkit_dom_html_table_col_element_set_span:
 * @self: A #WebKitDOMHTMLTableColElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_col_element_set_span(WebKitDOMHTMLTableColElement* self, glong value);

/**
 * webkit_dom_html_table_col_element_get_v_align:
 * @self: A #WebKitDOMHTMLTableColElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_col_element_get_v_align(WebKitDOMHTMLTableColElement* self);

/**
 * webkit_dom_html_table_col_element_set_v_align:
 * @self: A #WebKitDOMHTMLTableColElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_col_element_set_v_align(WebKitDOMHTMLTableColElement* self, const gchar* value);

/**
 * webkit_dom_html_table_col_element_get_width:
 * @self: A #WebKitDOMHTMLTableColElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_col_element_get_width(WebKitDOMHTMLTableColElement* self);

/**
 * webkit_dom_html_table_col_element_set_width:
 * @self: A #WebKitDOMHTMLTableColElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_col_element_set_width(WebKitDOMHTMLTableColElement* self, const gchar* value);

G_END_DECLS

#endif /* WebKitDOMHTMLTableColElement_h */
