/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
#ifndef WebKitDOMEventTarget_h
#define WebKitDOMEventTarget_h

#include <glib-object.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_EVENT_TARGET            (webkit_dom_event_target_get_type ())
#define WEBKIT_DOM_EVENT_TARGET(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_DOM_TYPE_EVENT_TARGET, WebKitDOMEventTarget))
#define WEBKIT_DOM_EVENT_TARGET_CLASS(obj)      (G_TYPE_CHECK_CLASS_CAST ((obj), WEBKIT_DOM_TYPE_EVENT_TARGET, WebKitDOMEventTargetIface))
#define WEBKIT_DOM_IS_EVENT_TARGET(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_DOM_TYPE_EVENT_TARGET))
#define WEBKIT_DOM_EVENT_TARGET_GET_IFACE(obj)  (G_TYPE_INSTANCE_GET_INTERFACE ((obj), WEBKIT_DOM_TYPE_EVENT_TARGET, WebKitDOMEventTargetIface))

struct _WebKitDOMEventTargetIface {
    GTypeInterface gIface;

    /* virtual table */
    gboolean      (* dispatch_event)(WebKitDOMEventTarget *target,
                                     WebKitDOMEvent       *event,
                                     GError              **error);

    gboolean      (* add_event_listener)(WebKitDOMEventTarget *target,
                                         const char           *event_name,
                                         GClosure             *handler,
                                         gboolean              use_capture);
    gboolean      (* remove_event_listener)(WebKitDOMEventTarget *target,
                                            const char           *event_name,
                                            GClosure             *handler,
                                            gboolean              use_capture);

    void (*_webkitdom_reserved0) (void);
    void (*_webkitdom_reserved1) (void);
    void (*_webkitdom_reserved2) (void);
    void (*_webkitdom_reserved3) (void);
};


WEBKIT_DEPRECATED GType     webkit_dom_event_target_get_type(void) G_GNUC_CONST;

/**
 * webkit_dom_event_target_dispatch_event:
 * @target: A #WebKitDOMEventTarget
 * @event: A #WebKitDOMEvent
 * @error: return location for an error or %NULL
 *
 * Returns: a #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean  webkit_dom_event_target_dispatch_event(WebKitDOMEventTarget *target,
                                                            WebKitDOMEvent       *event,
                                                            GError              **error);

/**
 * webkit_dom_event_target_add_event_listener:
 * @target: A #WebKitDOMEventTarget
 * @event_name: A #gchar
 * @handler: (scope async): A #GCallback
 * @use_capture: A #gboolean
 * @user_data: A #gpointer
 *
 * Returns: a #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean  webkit_dom_event_target_add_event_listener(WebKitDOMEventTarget *target,
                                                                const char           *event_name,
                                                                GCallback             handler,
                                                                gboolean              use_capture,
                                                                gpointer              user_data);

/**
 * webkit_dom_event_target_remove_event_listener:
 * @target: A #WebKitDOMEventTarget
 * @event_name: A #gchar
 * @handler: (type gpointer): A #GCallback
 * @use_capture: A #gboolean
 *
 * Returns: a #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean  webkit_dom_event_target_remove_event_listener(WebKitDOMEventTarget *target,
                                                                   const char           *event_name,
                                                                   GCallback             handler,
                                                                   gboolean              use_capture);

/**
 * webkit_dom_event_target_add_event_listener_with_closure: (rename-to webkit_dom_event_target_add_event_listener)
 * @target: A #WebKitDOMEventTarget
 * @event_name: A #gchar
 * @handler: A #GClosure
 * @use_capture: A #gboolean
 *
 * Version of webkit_dom_event_target_add_event_listener() using a closure
 * instead of a callbacks for easier binding in other languages.
 *
 * Returns: a #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean webkit_dom_event_target_add_event_listener_with_closure(WebKitDOMEventTarget *target,
                                                                            const char           *event_name,
                                                                            GClosure             *handler,
                                                                            gboolean              use_capture);

/**
 * webkit_dom_event_target_remove_event_listener_with_closure: (rename-to webkit_dom_event_target_remove_event_listener)
 * @target: A #WebKitDOMEventTarget
 * @event_name: A #gchar
 * @handler: A #GClosure
 * @use_capture: A #gboolean
 *
 * Version of webkit_dom_event_target_remove_event_listener() using a closure
 * instead of a callbacks for easier binding in other languages.
 *
 * Returns: a #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gboolean webkit_dom_event_target_remove_event_listener_with_closure(WebKitDOMEventTarget *target,
                                                                               const char           *event_name,
                                                                               GClosure             *handler,
                                                                               gboolean              use_capture);


G_END_DECLS

#endif /* WebKitDOMEventTarget_h */
