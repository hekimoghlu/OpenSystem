/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#ifndef WebKitDOMMediaList_h
#define WebKitDOMMediaList_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMObject.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_MEDIA_LIST            (webkit_dom_media_list_get_type())
#define WEBKIT_DOM_MEDIA_LIST(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_MEDIA_LIST, WebKitDOMMediaList))
#define WEBKIT_DOM_MEDIA_LIST_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_MEDIA_LIST, WebKitDOMMediaListClass)
#define WEBKIT_DOM_IS_MEDIA_LIST(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_MEDIA_LIST))
#define WEBKIT_DOM_IS_MEDIA_LIST_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_MEDIA_LIST))
#define WEBKIT_DOM_MEDIA_LIST_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_MEDIA_LIST, WebKitDOMMediaListClass))

struct _WebKitDOMMediaList {
    WebKitDOMObject parent_instance;
};

struct _WebKitDOMMediaListClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_media_list_get_type(void);

/**
 * webkit_dom_media_list_item:
 * @self: A #WebKitDOMMediaList
 * @index: A #gulong
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_media_list_item(WebKitDOMMediaList* self, gulong index);

/**
 * webkit_dom_media_list_delete_medium:
 * @self: A #WebKitDOMMediaList
 * @oldMedium: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_media_list_delete_medium(WebKitDOMMediaList* self, const gchar* oldMedium, GError** error);

/**
 * webkit_dom_media_list_append_medium:
 * @self: A #WebKitDOMMediaList
 * @newMedium: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_media_list_append_medium(WebKitDOMMediaList* self, const gchar* newMedium, GError** error);

/**
 * webkit_dom_media_list_get_media_text:
 * @self: A #WebKitDOMMediaList
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_media_list_get_media_text(WebKitDOMMediaList* self);

/**
 * webkit_dom_media_list_set_media_text:
 * @self: A #WebKitDOMMediaList
 * @value: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_media_list_set_media_text(WebKitDOMMediaList* self, const gchar* value, GError** error);

/**
 * webkit_dom_media_list_get_length:
 * @self: A #WebKitDOMMediaList
 *
 * Returns: A #gulong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gulong
webkit_dom_media_list_get_length(WebKitDOMMediaList* self);

G_END_DECLS

#endif /* WebKitDOMMediaList_h */
