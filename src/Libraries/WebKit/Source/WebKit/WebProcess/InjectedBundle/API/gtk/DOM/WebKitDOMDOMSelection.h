/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#if !defined(__WEBKITDOM_H_INSIDE__) && !defined(BUILDING_WEBKIT) && !defined(WEBKIT_DOM_USE_UNSTABLE_API)
#error "Only <webkitdom/webkitdom.h> can be included directly."
#endif

#ifndef WebKitDOMDOMSelection_h
#define WebKitDOMDOMSelection_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMObject.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_DOM_SELECTION            (webkit_dom_dom_selection_get_type())
#define WEBKIT_DOM_DOM_SELECTION(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_DOM_SELECTION, WebKitDOMDOMSelection))
#define WEBKIT_DOM_DOM_SELECTION_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_DOM_SELECTION, WebKitDOMDOMSelectionClass)
#define WEBKIT_DOM_IS_DOM_SELECTION(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_DOM_SELECTION))
#define WEBKIT_DOM_IS_DOM_SELECTION_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_DOM_SELECTION))
#define WEBKIT_DOM_DOM_SELECTION_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_DOM_SELECTION, WebKitDOMDOMSelectionClass))

struct _WebKitDOMDOMSelection {
    WebKitDOMObject parent_instance;
};

struct _WebKitDOMDOMSelectionClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_dom_selection_get_type(void);

/**
 * webkit_dom_dom_selection_collapse:
 * @self: A #WebKitDOMDOMSelection
 * @node: A #WebKitDOMNode
 * @offset: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_collapse(WebKitDOMDOMSelection* self, WebKitDOMNode* node, gulong offset);

/**
 * webkit_dom_dom_selection_collapse_to_end:
 * @self: A #WebKitDOMDOMSelection
 * @error: #GError
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_collapse_to_end(WebKitDOMDOMSelection* self, GError** error);

/**
 * webkit_dom_dom_selection_collapse_to_start:
 * @self: A #WebKitDOMDOMSelection
 * @error: #GError
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_collapse_to_start(WebKitDOMDOMSelection* self, GError** error);

/**
 * webkit_dom_dom_selection_delete_from_document:
 * @self: A #WebKitDOMDOMSelection
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_delete_from_document(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_contains_node:
 * @self: A #WebKitDOMDOMSelection
 * @node: A #WebKitDOMNode
 * @allowPartial: A #gboolean
 *
 * Returns: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean
webkit_dom_dom_selection_contains_node(WebKitDOMDOMSelection* self, WebKitDOMNode* node, gboolean allowPartial);

/**
 * webkit_dom_dom_selection_select_all_children:
 * @self: A #WebKitDOMDOMSelection
 * @node: A #WebKitDOMNode
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_select_all_children(WebKitDOMDOMSelection* self, WebKitDOMNode* node);

/**
 * webkit_dom_dom_selection_extend:
 * @self: A #WebKitDOMDOMSelection
 * @node: A #WebKitDOMNode
 * @offset: A #gulong
 * @error: #GError
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_extend(WebKitDOMDOMSelection* self, WebKitDOMNode* node, gulong offset, GError** error);

/**
 * webkit_dom_dom_selection_get_range_at:
 * @self: A #WebKitDOMDOMSelection
 * @index: A #gulong
 * @error: #GError
 *
 * Returns: (transfer full): A #WebKitDOMRange
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMRange*
webkit_dom_dom_selection_get_range_at(WebKitDOMDOMSelection* self, gulong index, GError** error);

/**
 * webkit_dom_dom_selection_remove_all_ranges:
 * @self: A #WebKitDOMDOMSelection
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_remove_all_ranges(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_add_range:
 * @self: A #WebKitDOMDOMSelection
 * @range: A #WebKitDOMRange
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_add_range(WebKitDOMDOMSelection* self, WebKitDOMRange* range);

/**
 * webkit_dom_dom_selection_set_base_and_extent:
 * @self: A #WebKitDOMDOMSelection
 * @baseNode: A #WebKitDOMNode
 * @baseOffset: A #gulong
 * @extentNode: A #WebKitDOMNode
 * @extentOffset: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_set_base_and_extent(WebKitDOMDOMSelection* self, WebKitDOMNode* baseNode, gulong baseOffset, WebKitDOMNode* extentNode, gulong extentOffset);

/**
 * webkit_dom_dom_selection_set_position:
 * @self: A #WebKitDOMDOMSelection
 * @node: A #WebKitDOMNode
 * @offset: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_set_position(WebKitDOMDOMSelection* self, WebKitDOMNode* node, gulong offset);

/**
 * webkit_dom_dom_selection_empty:
 * @self: A #WebKitDOMDOMSelection
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_empty(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_modify:
 * @self: A #WebKitDOMDOMSelection
 * @alter: A #gchar
 * @direction: A #gchar
 * @granularity: A #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_selection_modify(WebKitDOMDOMSelection* self, const gchar* alter, const gchar* direction, const gchar* granularity);

/**
 * webkit_dom_dom_selection_get_anchor_node:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_dom_selection_get_anchor_node(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_anchor_offset:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gulong
webkit_dom_dom_selection_get_anchor_offset(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_focus_node:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_dom_selection_get_focus_node(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_focus_offset:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gulong
webkit_dom_dom_selection_get_focus_offset(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_is_collapsed:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean
webkit_dom_dom_selection_get_is_collapsed(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_range_count:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gulong
webkit_dom_dom_selection_get_range_count(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_selection_type:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: A #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gchar*
webkit_dom_dom_selection_get_selection_type(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_base_node:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_dom_selection_get_base_node(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_base_offset:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gulong
webkit_dom_dom_selection_get_base_offset(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_extent_node:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_dom_selection_get_extent_node(WebKitDOMDOMSelection* self);

/**
 * webkit_dom_dom_selection_get_extent_offset:
 * @self: A #WebKitDOMDOMSelection
 *
 * Returns: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gulong
webkit_dom_dom_selection_get_extent_offset(WebKitDOMDOMSelection* self);

G_END_DECLS

#endif /* WebKitDOMDOMSelection_h */
