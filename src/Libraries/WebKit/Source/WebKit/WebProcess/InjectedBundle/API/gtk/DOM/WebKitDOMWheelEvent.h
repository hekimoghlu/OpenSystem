/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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

#ifndef WebKitDOMWheelEvent_h
#define WebKitDOMWheelEvent_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMMouseEvent.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_WHEEL_EVENT            (webkit_dom_wheel_event_get_type())
#define WEBKIT_DOM_WHEEL_EVENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_WHEEL_EVENT, WebKitDOMWheelEvent))
#define WEBKIT_DOM_WHEEL_EVENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_WHEEL_EVENT, WebKitDOMWheelEventClass)
#define WEBKIT_DOM_IS_WHEEL_EVENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_WHEEL_EVENT))
#define WEBKIT_DOM_IS_WHEEL_EVENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_WHEEL_EVENT))
#define WEBKIT_DOM_WHEEL_EVENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_WHEEL_EVENT, WebKitDOMWheelEventClass))

struct _WebKitDOMWheelEvent {
    WebKitDOMMouseEvent parent_instance;
};

struct _WebKitDOMWheelEventClass {
    WebKitDOMMouseEventClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_wheel_event_get_type(void);

/**
 * webkit_dom_wheel_event_init_wheel_event:
 * @self: A #WebKitDOMWheelEvent
 * @wheelDeltaX: A #glong
 * @wheelDeltaY: A #glong
 * @view: A #WebKitDOMDOMWindow
 * @screenX: A #glong
 * @screenY: A #glong
 * @clientX: A #glong
 * @clientY: A #glong
 * @ctrlKey: A #gboolean
 * @altKey: A #gboolean
 * @shiftKey: A #gboolean
 * @metaKey: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_wheel_event_init_wheel_event(WebKitDOMWheelEvent* self, glong wheelDeltaX, glong wheelDeltaY, WebKitDOMDOMWindow* view, glong screenX, glong screenY, glong clientX, glong clientY, gboolean ctrlKey, gboolean altKey, gboolean shiftKey, gboolean metaKey);

/**
 * webkit_dom_wheel_event_get_wheel_delta_x:
 * @self: A #WebKitDOMWheelEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_wheel_event_get_wheel_delta_x(WebKitDOMWheelEvent* self);

/**
 * webkit_dom_wheel_event_get_wheel_delta_y:
 * @self: A #WebKitDOMWheelEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_wheel_event_get_wheel_delta_y(WebKitDOMWheelEvent* self);

/**
 * webkit_dom_wheel_event_get_wheel_delta:
 * @self: A #WebKitDOMWheelEvent
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_wheel_event_get_wheel_delta(WebKitDOMWheelEvent* self);

G_END_DECLS

#endif /* WebKitDOMWheelEvent_h */
