/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

#ifndef WebKitDOMDOMTokenList_h
#define WebKitDOMDOMTokenList_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMObject.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_DOM_TOKEN_LIST            (webkit_dom_dom_token_list_get_type())
#define WEBKIT_DOM_DOM_TOKEN_LIST(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_DOM_TOKEN_LIST, WebKitDOMDOMTokenList))
#define WEBKIT_DOM_DOM_TOKEN_LIST_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_DOM_TOKEN_LIST, WebKitDOMDOMTokenListClass)
#define WEBKIT_DOM_IS_DOM_TOKEN_LIST(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_DOM_TOKEN_LIST))
#define WEBKIT_DOM_IS_DOM_TOKEN_LIST_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_DOM_TOKEN_LIST))
#define WEBKIT_DOM_DOM_TOKEN_LIST_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_DOM_TOKEN_LIST, WebKitDOMDOMTokenListClass))

struct _WebKitDOMDOMTokenList {
    WebKitDOMObject parent_instance;
};

struct _WebKitDOMDOMTokenListClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_dom_token_list_get_type(void);

/**
 * webkit_dom_dom_token_list_item:
 * @self: A #WebKitDOMDOMTokenList
 * @index: A #gulong
 *
 * Returns: A #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gchar*
webkit_dom_dom_token_list_item(WebKitDOMDOMTokenList* self, gulong index);

/**
 * webkit_dom_dom_token_list_contains:
 * @self: A #WebKitDOMDOMTokenList
 * @token: A #gchar
 *
 * Returns: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean
webkit_dom_dom_token_list_contains(WebKitDOMDOMTokenList* self, const gchar* token);

/**
 * webkit_dom_dom_token_list_add:
 * @self: A #WebKitDOMDOMTokenList
 * @error: #GError
 * @...: list of #gchar ended by %NULL.
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_token_list_add(WebKitDOMDOMTokenList* self, GError** error, ...);

/**
 * webkit_dom_dom_token_list_remove:
 * @self: A #WebKitDOMDOMTokenList
 * @error: #GError
 * @...: list of #gchar ended by %NULL.
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_token_list_remove(WebKitDOMDOMTokenList* self, GError** error, ...);

/**
 * webkit_dom_dom_token_list_toggle:
 * @self: A #WebKitDOMDOMTokenList
 * @token: A #gchar
 * @force: A #gboolean
 * @error: #GError
 *
 * Returns: A #gboolean
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean
webkit_dom_dom_token_list_toggle(WebKitDOMDOMTokenList* self, const gchar* token, gboolean force, GError** error);

/**
 * webkit_dom_dom_token_list_replace:
 * @self: A #WebKitDOMDOMTokenList
 * @token: A #gchar
 * @newToken: A #gchar
 * @error: #GError
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_token_list_replace(WebKitDOMDOMTokenList* self, const gchar* token, const gchar* newToken, GError** error);

/**
 * webkit_dom_dom_token_list_get_length:
 * @self: A #WebKitDOMDOMTokenList
 *
 * Returns: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gulong
webkit_dom_dom_token_list_get_length(WebKitDOMDOMTokenList* self);

/**
 * webkit_dom_dom_token_list_get_value:
 * @self: A #WebKitDOMDOMTokenList
 *
 * Returns: A #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gchar*
webkit_dom_dom_token_list_get_value(WebKitDOMDOMTokenList* self);

/**
 * webkit_dom_dom_token_list_set_value:
 * @self: A #WebKitDOMDOMTokenList
 * @value: A #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED void
webkit_dom_dom_token_list_set_value(WebKitDOMDOMTokenList* self, const gchar* value);

G_END_DECLS

#endif /* WebKitDOMDOMTokenList_h */
