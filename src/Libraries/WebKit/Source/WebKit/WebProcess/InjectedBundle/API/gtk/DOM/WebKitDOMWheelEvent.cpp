/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include "WebKitDOMWheelEvent.h"

#include <WebCore/CSSImportRule.h>
#include "DOMObjectCache.h"
#include <WebCore/Document.h>
#include <WebCore/ExceptionCode.h>
#include <WebCore/JSExecState.h>
#include "WebKitDOMDOMWindowPrivate.h"
#include "WebKitDOMEventPrivate.h"
#include "WebKitDOMPrivate.h"
#include "WebKitDOMWheelEventPrivate.h"
#include "ConvertToUTF8String.h"
#include <wtf/GetPtr.h>
#include <wtf/RefPtr.h>

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

WebKitDOMWheelEvent* kit(WebCore::WheelEvent* obj)
{
    return WEBKIT_DOM_WHEEL_EVENT(kit(static_cast<WebCore::Event*>(obj)));
}

WebCore::WheelEvent* core(WebKitDOMWheelEvent* request)
{
    return request ? static_cast<WebCore::WheelEvent*>(WEBKIT_DOM_OBJECT(request)->coreObject) : 0;
}

WebKitDOMWheelEvent* wrapWheelEvent(WebCore::WheelEvent* coreObject)
{
    ASSERT(coreObject);
    return WEBKIT_DOM_WHEEL_EVENT(g_object_new(WEBKIT_DOM_TYPE_WHEEL_EVENT, "core-object", coreObject, nullptr));
}

} // namespace WebKit

G_DEFINE_TYPE(WebKitDOMWheelEvent, webkit_dom_wheel_event, WEBKIT_DOM_TYPE_MOUSE_EVENT)

enum {
    DOM_WHEEL_EVENT_PROP_0,
    DOM_WHEEL_EVENT_PROP_WHEEL_DELTA_X,
    DOM_WHEEL_EVENT_PROP_WHEEL_DELTA_Y,
    DOM_WHEEL_EVENT_PROP_WHEEL_DELTA,
};

static void webkit_dom_wheel_event_get_property(GObject* object, guint propertyId, GValue* value, GParamSpec* pspec)
{
    WebKitDOMWheelEvent* self = WEBKIT_DOM_WHEEL_EVENT(object);

    switch (propertyId) {
    case DOM_WHEEL_EVENT_PROP_WHEEL_DELTA_X:
        g_value_set_long(value, webkit_dom_wheel_event_get_wheel_delta_x(self));
        break;
    case DOM_WHEEL_EVENT_PROP_WHEEL_DELTA_Y:
        g_value_set_long(value, webkit_dom_wheel_event_get_wheel_delta_y(self));
        break;
    case DOM_WHEEL_EVENT_PROP_WHEEL_DELTA:
        g_value_set_long(value, webkit_dom_wheel_event_get_wheel_delta(self));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}

static void webkit_dom_wheel_event_class_init(WebKitDOMWheelEventClass* requestClass)
{
    GObjectClass* gobjectClass = G_OBJECT_CLASS(requestClass);
    gobjectClass->get_property = webkit_dom_wheel_event_get_property;

    g_object_class_install_property(
        gobjectClass,
        DOM_WHEEL_EVENT_PROP_WHEEL_DELTA_X,
        g_param_spec_long(
            "wheel-delta-x",
            "WheelEvent:wheel-delta-x",
            "read-only glong WheelEvent:wheel-delta-x",
            G_MINLONG, G_MAXLONG, 0,
            WEBKIT_PARAM_READABLE));

    g_object_class_install_property(
        gobjectClass,
        DOM_WHEEL_EVENT_PROP_WHEEL_DELTA_Y,
        g_param_spec_long(
            "wheel-delta-y",
            "WheelEvent:wheel-delta-y",
            "read-only glong WheelEvent:wheel-delta-y",
            G_MINLONG, G_MAXLONG, 0,
            WEBKIT_PARAM_READABLE));

    g_object_class_install_property(
        gobjectClass,
        DOM_WHEEL_EVENT_PROP_WHEEL_DELTA,
        g_param_spec_long(
            "wheel-delta",
            "WheelEvent:wheel-delta",
            "read-only glong WheelEvent:wheel-delta",
            G_MINLONG, G_MAXLONG, 0,
            WEBKIT_PARAM_READABLE));
}

static void webkit_dom_wheel_event_init(WebKitDOMWheelEvent* request)
{
    UNUSED_PARAM(request);
}

void webkit_dom_wheel_event_init_wheel_event(WebKitDOMWheelEvent* self, glong wheelDeltaX, glong wheelDeltaY, WebKitDOMDOMWindow* view, glong screenX, glong screenY, glong clientX, glong clientY, gboolean ctrlKey, gboolean altKey, gboolean shiftKey, gboolean metaKey)
{
    WebCore::JSMainThreadNullState state;
    g_return_if_fail(WEBKIT_DOM_IS_WHEEL_EVENT(self));
    g_return_if_fail(WEBKIT_DOM_IS_DOM_WINDOW(view));
}

glong webkit_dom_wheel_event_get_wheel_delta_x(WebKitDOMWheelEvent* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_WHEEL_EVENT(self), 0);
    WebCore::WheelEvent* item = WebKit::core(self);
    glong result = item->wheelDeltaX();
    return result;
}

glong webkit_dom_wheel_event_get_wheel_delta_y(WebKitDOMWheelEvent* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_WHEEL_EVENT(self), 0);
    WebCore::WheelEvent* item = WebKit::core(self);
    glong result = item->wheelDeltaY();
    return result;
}

glong webkit_dom_wheel_event_get_wheel_delta(WebKitDOMWheelEvent* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_WHEEL_EVENT(self), 0);
    WebCore::WheelEvent* item = WebKit::core(self);
    glong result = item->wheelDelta();
    return result;
}
G_GNUC_END_IGNORE_DEPRECATIONS;
