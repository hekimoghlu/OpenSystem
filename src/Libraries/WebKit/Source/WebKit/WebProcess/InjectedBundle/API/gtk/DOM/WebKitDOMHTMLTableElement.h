/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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

#ifndef WebKitDOMHTMLTableElement_h
#define WebKitDOMHTMLTableElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_TABLE_ELEMENT            (webkit_dom_html_table_element_get_type())
#define WEBKIT_DOM_HTML_TABLE_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_TABLE_ELEMENT, WebKitDOMHTMLTableElement))
#define WEBKIT_DOM_HTML_TABLE_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_TABLE_ELEMENT, WebKitDOMHTMLTableElementClass)
#define WEBKIT_DOM_IS_HTML_TABLE_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_TABLE_ELEMENT))
#define WEBKIT_DOM_IS_HTML_TABLE_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_TABLE_ELEMENT))
#define WEBKIT_DOM_HTML_TABLE_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_TABLE_ELEMENT, WebKitDOMHTMLTableElementClass))

struct _WebKitDOMHTMLTableElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLTableElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_table_element_get_type(void);

/**
 * webkit_dom_html_table_element_create_t_head:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: (transfer none): A #WebKitDOMHTMLElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLElement*
webkit_dom_html_table_element_create_t_head(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_delete_t_head:
 * @self: A #WebKitDOMHTMLTableElement
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_delete_t_head(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_create_t_foot:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: (transfer none): A #WebKitDOMHTMLElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLElement*
webkit_dom_html_table_element_create_t_foot(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_delete_t_foot:
 * @self: A #WebKitDOMHTMLTableElement
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_delete_t_foot(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_create_caption:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: (transfer none): A #WebKitDOMHTMLElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLElement*
webkit_dom_html_table_element_create_caption(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_delete_caption:
 * @self: A #WebKitDOMHTMLTableElement
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_delete_caption(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_insert_row:
 * @self: A #WebKitDOMHTMLTableElement
 * @index: A #glong
 * @error: #GError
 *
 * Returns: (transfer none): A #WebKitDOMHTMLElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLElement*
webkit_dom_html_table_element_insert_row(WebKitDOMHTMLTableElement* self, glong index, GError** error);

/**
 * webkit_dom_html_table_element_delete_row:
 * @self: A #WebKitDOMHTMLTableElement
 * @index: A #glong
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_delete_row(WebKitDOMHTMLTableElement* self, glong index, GError** error);

/**
 * webkit_dom_html_table_element_get_caption:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: (transfer none): A #WebKitDOMHTMLTableCaptionElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLTableCaptionElement*
webkit_dom_html_table_element_get_caption(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_caption:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #WebKitDOMHTMLTableCaptionElement
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_caption(WebKitDOMHTMLTableElement* self, WebKitDOMHTMLTableCaptionElement* value, GError** error);

/**
 * webkit_dom_html_table_element_get_t_head:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: (transfer none): A #WebKitDOMHTMLTableSectionElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLTableSectionElement*
webkit_dom_html_table_element_get_t_head(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_t_head:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #WebKitDOMHTMLTableSectionElement
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_t_head(WebKitDOMHTMLTableElement* self, WebKitDOMHTMLTableSectionElement* value, GError** error);

/**
 * webkit_dom_html_table_element_get_t_foot:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: (transfer none): A #WebKitDOMHTMLTableSectionElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLTableSectionElement*
webkit_dom_html_table_element_get_t_foot(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_t_foot:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #WebKitDOMHTMLTableSectionElement
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_t_foot(WebKitDOMHTMLTableElement* self, WebKitDOMHTMLTableSectionElement* value, GError** error);

/**
 * webkit_dom_html_table_element_get_rows:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: (transfer full): A #WebKitDOMHTMLCollection
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLCollection*
webkit_dom_html_table_element_get_rows(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_get_t_bodies:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: (transfer full): A #WebKitDOMHTMLCollection
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLCollection*
webkit_dom_html_table_element_get_t_bodies(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_get_align:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_element_get_align(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_align:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_align(WebKitDOMHTMLTableElement* self, const gchar* value);

/**
 * webkit_dom_html_table_element_get_bg_color:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_element_get_bg_color(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_bg_color:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_bg_color(WebKitDOMHTMLTableElement* self, const gchar* value);

/**
 * webkit_dom_html_table_element_get_border:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_element_get_border(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_border:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_border(WebKitDOMHTMLTableElement* self, const gchar* value);

/**
 * webkit_dom_html_table_element_get_cell_padding:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_element_get_cell_padding(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_cell_padding:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_cell_padding(WebKitDOMHTMLTableElement* self, const gchar* value);

/**
 * webkit_dom_html_table_element_get_cell_spacing:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_element_get_cell_spacing(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_cell_spacing:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_cell_spacing(WebKitDOMHTMLTableElement* self, const gchar* value);

/**
 * webkit_dom_html_table_element_get_rules:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_element_get_rules(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_rules:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_rules(WebKitDOMHTMLTableElement* self, const gchar* value);

/**
 * webkit_dom_html_table_element_get_summary:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_element_get_summary(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_summary:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_summary(WebKitDOMHTMLTableElement* self, const gchar* value);

/**
 * webkit_dom_html_table_element_get_width:
 * @self: A #WebKitDOMHTMLTableElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_table_element_get_width(WebKitDOMHTMLTableElement* self);

/**
 * webkit_dom_html_table_element_set_width:
 * @self: A #WebKitDOMHTMLTableElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_table_element_set_width(WebKitDOMHTMLTableElement* self, const gchar* value);

G_END_DECLS

#endif /* WebKitDOMHTMLTableElement_h */
