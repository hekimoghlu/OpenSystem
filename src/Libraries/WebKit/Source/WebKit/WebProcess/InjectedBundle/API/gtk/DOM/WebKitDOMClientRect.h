/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#ifndef WebKitDOMClientRect_h
#define WebKitDOMClientRect_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMObject.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_CLIENT_RECT            (webkit_dom_client_rect_get_type())
#define WEBKIT_DOM_CLIENT_RECT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_CLIENT_RECT, WebKitDOMClientRect))
#define WEBKIT_DOM_CLIENT_RECT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_CLIENT_RECT, WebKitDOMClientRectClass)
#define WEBKIT_DOM_IS_CLIENT_RECT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_CLIENT_RECT))
#define WEBKIT_DOM_IS_CLIENT_RECT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_CLIENT_RECT))
#define WEBKIT_DOM_CLIENT_RECT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_CLIENT_RECT, WebKitDOMClientRectClass))

struct _WebKitDOMClientRect {
    WebKitDOMObject parent_instance;
};

struct _WebKitDOMClientRectClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_client_rect_get_type(void);

/**
 * webkit_dom_client_rect_get_top:
 * @self: A #WebKitDOMClientRect
 *
 * Returns the top coordinate of @self, relative to the viewport.
 *
 * Returns: A #gfloat
 *
 * Since: 2.18
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gfloat
webkit_dom_client_rect_get_top(WebKitDOMClientRect* self);

/**
 * webkit_dom_client_rect_get_right:
 * @self: A #WebKitDOMClientRect
 *
 * Returns the right coordinate of @self, relative to the viewport.
 *
 * Returns: A #gfloat
 *
 * Since: 2.18
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gfloat
webkit_dom_client_rect_get_right(WebKitDOMClientRect* self);

/**
 * webkit_dom_client_rect_get_bottom:
 * @self: A #WebKitDOMClientRect
 *
 * Returns the bottom coordinate of @self, relative to the viewport.
 *
 * Returns: A #gfloat
 *
 * Since: 2.18
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gfloat
webkit_dom_client_rect_get_bottom(WebKitDOMClientRect* self);

/**
 * webkit_dom_client_rect_get_left:
 * @self: A #WebKitDOMClientRect
 *
 * Returns the left coordinate of @self, relative to the viewport.
 *
 * Returns: A #gfloat
 *
 * Since: 2.18
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gfloat
webkit_dom_client_rect_get_left(WebKitDOMClientRect* self);

/**
 * webkit_dom_client_rect_get_width:
 * @self: A #WebKitDOMClientRect
 *
 * Returns the width of @self.
 *
 * Returns: A #gfloat
 *
 * Since: 2.18
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gfloat
webkit_dom_client_rect_get_width(WebKitDOMClientRect* self);

/**
 * webkit_dom_client_rect_get_height:
 * @self: A #WebKitDOMClientRect
 *
 * Returns the height of @self.
 *
 * Returns: A #gfloat
 *
 * Since: 2.18
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gfloat
webkit_dom_client_rect_get_height(WebKitDOMClientRect* self);

G_END_DECLS

#endif /* WebKitDOMClientRect_h */
