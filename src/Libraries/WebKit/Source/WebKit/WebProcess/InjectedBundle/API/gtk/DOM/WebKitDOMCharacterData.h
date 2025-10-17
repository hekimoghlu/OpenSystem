/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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

#ifndef WebKitDOMCharacterData_h
#define WebKitDOMCharacterData_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMNode.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_CHARACTER_DATA            (webkit_dom_character_data_get_type())
#define WEBKIT_DOM_CHARACTER_DATA(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_CHARACTER_DATA, WebKitDOMCharacterData))
#define WEBKIT_DOM_CHARACTER_DATA_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_CHARACTER_DATA, WebKitDOMCharacterDataClass)
#define WEBKIT_DOM_IS_CHARACTER_DATA(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_CHARACTER_DATA))
#define WEBKIT_DOM_IS_CHARACTER_DATA_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_CHARACTER_DATA))
#define WEBKIT_DOM_CHARACTER_DATA_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_CHARACTER_DATA, WebKitDOMCharacterDataClass))

struct _WebKitDOMCharacterData {
    WebKitDOMNode parent_instance;
};

struct _WebKitDOMCharacterDataClass {
    WebKitDOMNodeClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_character_data_get_type(void);

/**
 * webkit_dom_character_data_substring_data:
 * @self: A #WebKitDOMCharacterData
 * @offset: A #gulong
 * @length: A #gulong
 * @error: #GError
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_character_data_substring_data(WebKitDOMCharacterData* self, gulong offset, gulong length, GError** error);

/**
 * webkit_dom_character_data_append_data:
 * @self: A #WebKitDOMCharacterData
 * @data: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_character_data_append_data(WebKitDOMCharacterData* self, const gchar* data, GError** error);

/**
 * webkit_dom_character_data_insert_data:
 * @self: A #WebKitDOMCharacterData
 * @offset: A #gulong
 * @data: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_character_data_insert_data(WebKitDOMCharacterData* self, gulong offset, const gchar* data, GError** error);

/**
 * webkit_dom_character_data_delete_data:
 * @self: A #WebKitDOMCharacterData
 * @offset: A #gulong
 * @length: A #gulong
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_character_data_delete_data(WebKitDOMCharacterData* self, gulong offset, gulong length, GError** error);

/**
 * webkit_dom_character_data_replace_data:
 * @self: A #WebKitDOMCharacterData
 * @offset: A #gulong
 * @length: A #gulong
 * @data: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_character_data_replace_data(WebKitDOMCharacterData* self, gulong offset, gulong length, const gchar* data, GError** error);

/**
 * webkit_dom_character_data_get_data:
 * @self: A #WebKitDOMCharacterData
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_character_data_get_data(WebKitDOMCharacterData* self);

/**
 * webkit_dom_character_data_set_data:
 * @self: A #WebKitDOMCharacterData
 * @value: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_character_data_set_data(WebKitDOMCharacterData* self, const gchar* value, GError** error);

/**
 * webkit_dom_character_data_get_length:
 * @self: A #WebKitDOMCharacterData
 *
 * Returns: A #gulong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gulong
webkit_dom_character_data_get_length(WebKitDOMCharacterData* self);

G_END_DECLS

#endif /* WebKitDOMCharacterData_h */
