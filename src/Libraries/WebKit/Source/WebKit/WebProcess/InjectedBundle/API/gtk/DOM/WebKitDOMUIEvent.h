/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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

#ifndef WebKitDOMUIEvent_h
#define WebKitDOMUIEvent_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMEvent.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_UI_EVENT            (webkit_dom_ui_event_get_type())
#define WEBKIT_DOM_UI_EVENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_UI_EVENT, WebKitDOMUIEvent))
#define WEBKIT_DOM_UI_EVENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_UI_EVENT, WebKitDOMUIEventClass)
#define WEBKIT_DOM_IS_UI_EVENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_UI_EVENT))
#define WEBKIT_DOM_IS_UI_EVENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_UI_EVENT))
#define WEBKIT_DOM_UI_EVENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_UI_EVENT, WebKitDOMUIEventClass))

struct _WebKitDOMUIEvent {
    WebKitDOMEvent parent_instance;
};

struct _WebKitDOMUIEventClass {
    WebKitDOMEventClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_ui_event_get_type(void);

/**
 * webkit_dom_ui_event_init_ui_event:
 * @self: A #WebKitDOMUIEvent
 * @type: A #gchar
 * @canBubble: A #gboolean
 * @cancelable: A #gboolean
 * @view: A #WebKitDOMDOMWindow
 * @detail: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_ui_event_init_ui_event(WebKitDOMUIEvent* self, const gchar* type, gboolean canBubble, gboolean cancelable, WebKitDOMDOMWindow* view, glong detail);

/**
 * webkit_dom_ui_event_get_view:
 * @self: A #WebKitDOMUIEvent
 *
 * Returns: (transfer full): A #WebKitDOMDOMWindow
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMDOMWindow*
webkit_dom_ui_event_get_view(WebKitDOMUIEvent* self);

/**
 * webkit_dom_ui_event_get_detail:
 * @self: A #WebKitDOMUIEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_ui_event_get_detail(WebKitDOMUIEvent* self);

/**
 * webkit_dom_ui_event_get_key_code:
 * @self: A #WebKitDOMUIEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_ui_event_get_key_code(WebKitDOMUIEvent* self);

/**
 * webkit_dom_ui_event_get_char_code:
 * @self: A #WebKitDOMUIEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_ui_event_get_char_code(WebKitDOMUIEvent* self);

/**
 * webkit_dom_ui_event_get_layer_x:
 * @self: A #WebKitDOMUIEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_ui_event_get_layer_x(WebKitDOMUIEvent* self);

/**
 * webkit_dom_ui_event_get_layer_y:
 * @self: A #WebKitDOMUIEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_ui_event_get_layer_y(WebKitDOMUIEvent* self);

/**
 * webkit_dom_ui_event_get_page_x:
 * @self: A #WebKitDOMUIEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_ui_event_get_page_x(WebKitDOMUIEvent* self);

/**
 * webkit_dom_ui_event_get_page_y:
 * @self: A #WebKitDOMUIEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_ui_event_get_page_y(WebKitDOMUIEvent* self);

G_END_DECLS

#endif /* WebKitDOMUIEvent_h */
