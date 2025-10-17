/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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

#ifndef WebKitDOMKeyboardEvent_h
#define WebKitDOMKeyboardEvent_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMUIEvent.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_KEYBOARD_EVENT            (webkit_dom_keyboard_event_get_type())
#define WEBKIT_DOM_KEYBOARD_EVENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_KEYBOARD_EVENT, WebKitDOMKeyboardEvent))
#define WEBKIT_DOM_KEYBOARD_EVENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_KEYBOARD_EVENT, WebKitDOMKeyboardEventClass)
#define WEBKIT_DOM_IS_KEYBOARD_EVENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_KEYBOARD_EVENT))
#define WEBKIT_DOM_IS_KEYBOARD_EVENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_KEYBOARD_EVENT))
#define WEBKIT_DOM_KEYBOARD_EVENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_KEYBOARD_EVENT, WebKitDOMKeyboardEventClass))

#ifndef WEBKIT_DISABLE_DEPRECATED

/**
 * WEBKIT_DOM_KEYBOARD_EVENT_KEY_LOCATION_STANDARD:
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_KEYBOARD_EVENT_KEY_LOCATION_STANDARD 0x00

/**
 * WEBKIT_DOM_KEYBOARD_EVENT_KEY_LOCATION_LEFT:
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_KEYBOARD_EVENT_KEY_LOCATION_LEFT 0x01

/**
 * WEBKIT_DOM_KEYBOARD_EVENT_KEY_LOCATION_RIGHT:
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_KEYBOARD_EVENT_KEY_LOCATION_RIGHT 0x02

/**
 * WEBKIT_DOM_KEYBOARD_EVENT_KEY_LOCATION_NUMPAD:
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_KEYBOARD_EVENT_KEY_LOCATION_NUMPAD 0x03

#endif /* WEBKIT_DISABLE_DEPRECATED */

struct _WebKitDOMKeyboardEvent {
    WebKitDOMUIEvent parent_instance;
};

struct _WebKitDOMKeyboardEventClass {
    WebKitDOMUIEventClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_keyboard_event_get_type(void);

/**
 * webkit_dom_keyboard_event_get_modifier_state:
 * @self: A #WebKitDOMKeyboardEvent
 * @keyIdentifierArg: A #gchar
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_keyboard_event_get_modifier_state(WebKitDOMKeyboardEvent* self, const gchar* keyIdentifierArg);

/**
 * webkit_dom_keyboard_event_init_keyboard_event:
 * @self: A #WebKitDOMKeyboardEvent
 * @type: A #gchar
 * @canBubble: A #gboolean
 * @cancelable: A #gboolean
 * @view: A #WebKitDOMDOMWindow
 * @keyIdentifier: A #gchar
 * @location: A #gulong
 * @ctrlKey: A #gboolean
 * @altKey: A #gboolean
 * @shiftKey: A #gboolean
 * @metaKey: A #gboolean
 * @altGraphKey: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_keyboard_event_init_keyboard_event(WebKitDOMKeyboardEvent* self, const gchar* type, gboolean canBubble, gboolean cancelable, WebKitDOMDOMWindow* view, const gchar* keyIdentifier, gulong location, gboolean ctrlKey, gboolean altKey, gboolean shiftKey, gboolean metaKey, gboolean altGraphKey);

/**
 * webkit_dom_keyboard_event_get_key_identifier:
 * @self: A #WebKitDOMKeyboardEvent
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_keyboard_event_get_key_identifier(WebKitDOMKeyboardEvent* self);

/**
 * webkit_dom_keyboard_event_get_key_location:
 * @self: A #WebKitDOMKeyboardEvent
 *
 * Returns: A #gulong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gulong
webkit_dom_keyboard_event_get_key_location(WebKitDOMKeyboardEvent* self);

/**
 * webkit_dom_keyboard_event_get_ctrl_key:
 * @self: A #WebKitDOMKeyboardEvent
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_keyboard_event_get_ctrl_key(WebKitDOMKeyboardEvent* self);

/**
 * webkit_dom_keyboard_event_get_shift_key:
 * @self: A #WebKitDOMKeyboardEvent
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_keyboard_event_get_shift_key(WebKitDOMKeyboardEvent* self);

/**
 * webkit_dom_keyboard_event_get_alt_key:
 * @self: A #WebKitDOMKeyboardEvent
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_keyboard_event_get_alt_key(WebKitDOMKeyboardEvent* self);

/**
 * webkit_dom_keyboard_event_get_meta_key:
 * @self: A #WebKitDOMKeyboardEvent
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_keyboard_event_get_meta_key(WebKitDOMKeyboardEvent* self);

/**
 * webkit_dom_keyboard_event_get_alt_graph_key:
 * @self: A #WebKitDOMKeyboardEvent
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_keyboard_event_get_alt_graph_key(WebKitDOMKeyboardEvent* self);

G_END_DECLS

#endif /* WebKitDOMKeyboardEvent_h */
