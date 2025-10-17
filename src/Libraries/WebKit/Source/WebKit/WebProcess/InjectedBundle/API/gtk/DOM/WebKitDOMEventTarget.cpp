/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#include "config.h"
#include "WebKitDOMEventTarget.h"

#include "DOMObjectCache.h"
#include <WebCore/EventTarget.h>
#include "WebKitDOMEvent.h"
#include "WebKitDOMEventTargetPrivate.h"
#include "WebKitDOMPrivate.h"
#include <wtf/glib/GRefPtr.h>

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

typedef WebKitDOMEventTargetIface WebKitDOMEventTargetInterface;

G_DEFINE_INTERFACE(WebKitDOMEventTarget, webkit_dom_event_target, G_TYPE_OBJECT)

static void webkit_dom_event_target_default_init(WebKitDOMEventTargetIface*)
{
}

gboolean webkit_dom_event_target_dispatch_event(WebKitDOMEventTarget* target, WebKitDOMEvent* event, GError** error)
{
    g_return_val_if_fail(WEBKIT_DOM_IS_EVENT_TARGET(target), FALSE);
    g_return_val_if_fail(WEBKIT_DOM_IS_EVENT(event), FALSE);
    g_return_val_if_fail(!error || !*error, FALSE);

    return WEBKIT_DOM_EVENT_TARGET_GET_IFACE(target)->dispatch_event(target, event, error);
}

gboolean webkit_dom_event_target_add_event_listener(WebKitDOMEventTarget* target, const char* eventName, GCallback handler, gboolean useCapture, gpointer userData)
{

    g_return_val_if_fail(WEBKIT_DOM_IS_EVENT_TARGET(target), FALSE);
    g_return_val_if_fail(eventName, FALSE);

    GRefPtr<GClosure> closure = adoptGRef(g_cclosure_new(handler, userData, 0));
    return WEBKIT_DOM_EVENT_TARGET_GET_IFACE(target)->add_event_listener(target, eventName, closure.get(), useCapture);
}

gboolean webkit_dom_event_target_remove_event_listener(WebKitDOMEventTarget* target, const char* eventName, GCallback handler, gboolean useCapture)
{
    g_return_val_if_fail(WEBKIT_DOM_IS_EVENT_TARGET(target), FALSE);
    g_return_val_if_fail(eventName, FALSE);

    GRefPtr<GClosure> closure = adoptGRef(g_cclosure_new(handler, 0, 0));
    return WEBKIT_DOM_EVENT_TARGET_GET_IFACE(target)->remove_event_listener(target, eventName, closure.get(), useCapture);
}

gboolean webkit_dom_event_target_add_event_listener_with_closure(WebKitDOMEventTarget* target, const char* eventName, GClosure* handler, gboolean useCapture)
{
    g_return_val_if_fail(WEBKIT_DOM_IS_EVENT_TARGET(target), FALSE);
    g_return_val_if_fail(eventName, FALSE);
    g_return_val_if_fail(handler, FALSE);

    return WEBKIT_DOM_EVENT_TARGET_GET_IFACE(target)->add_event_listener(target, eventName, handler, useCapture);
}

gboolean webkit_dom_event_target_remove_event_listener_with_closure(WebKitDOMEventTarget* target, const char* eventName, GClosure* handler, gboolean useCapture)
{
    g_return_val_if_fail(WEBKIT_DOM_IS_EVENT_TARGET(target), FALSE);
    g_return_val_if_fail(eventName, FALSE);
    g_return_val_if_fail(handler, FALSE);

    return WEBKIT_DOM_EVENT_TARGET_GET_IFACE(target)->remove_event_listener(target, eventName, handler, useCapture);
}

namespace WebKit {

WebKitDOMEventTarget* kit(WebCore::EventTarget* obj)
{
    if (!obj)
        return 0;

    if (gpointer ret = DOMObjectCache::get(obj))
        return WEBKIT_DOM_EVENT_TARGET(ret);

    return wrap(obj);
}

WebCore::EventTarget* core(WebKitDOMEventTarget* request)
{
    return request ? static_cast<WebCore::EventTarget*>(WEBKIT_DOM_OBJECT(request)->coreObject) : 0;
}

} // namespace WebKit

G_GNUC_END_IGNORE_DEPRECATIONS;
