/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
#include "WebKitDOMHTMLMarqueeElement.h"

#include <WebCore/CSSImportRule.h>
#include "DOMObjectCache.h"
#include <WebCore/DOMException.h>
#include <WebCore/Document.h>
#include "GObjectEventListener.h"
#include <WebCore/HTMLNames.h>
#include <WebCore/JSExecState.h>
#include "WebKitDOMEventPrivate.h"
#include "WebKitDOMEventTarget.h"
#include "WebKitDOMHTMLMarqueeElementPrivate.h"
#include "WebKitDOMNodePrivate.h"
#include "WebKitDOMPrivate.h"
#include "ConvertToUTF8String.h"
#include <wtf/GetPtr.h>
#include <wtf/RefPtr.h>

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

WebKitDOMHTMLMarqueeElement* kit(WebCore::HTMLMarqueeElement* obj)
{
    return WEBKIT_DOM_HTML_MARQUEE_ELEMENT(kit(static_cast<WebCore::Node*>(obj)));
}

WebCore::HTMLMarqueeElement* core(WebKitDOMHTMLMarqueeElement* request)
{
    return request ? static_cast<WebCore::HTMLMarqueeElement*>(WEBKIT_DOM_OBJECT(request)->coreObject) : 0;
}

WebKitDOMHTMLMarqueeElement* wrapHTMLMarqueeElement(WebCore::HTMLMarqueeElement* coreObject)
{
    ASSERT(coreObject);
    return WEBKIT_DOM_HTML_MARQUEE_ELEMENT(g_object_new(WEBKIT_DOM_TYPE_HTML_MARQUEE_ELEMENT, "core-object", coreObject, nullptr));
}

} // namespace WebKit

static gboolean webkit_dom_html_marquee_element_dispatch_event(WebKitDOMEventTarget* target, WebKitDOMEvent* event, GError** error)
{
    WebCore::Event* coreEvent = WebKit::core(event);
    if (!coreEvent)
        return false;
    WebCore::HTMLMarqueeElement* coreTarget = static_cast<WebCore::HTMLMarqueeElement*>(WEBKIT_DOM_OBJECT(target)->coreObject);

    auto result = coreTarget->dispatchEventForBindings(*coreEvent);
    if (result.hasException()) {
        auto description = WebCore::DOMException::description(result.releaseException().code());
        g_set_error_literal(error, g_quark_from_string("WEBKIT_DOM"), description.legacyCode, description.name);
        return false;
    }
    return result.releaseReturnValue();
}

static gboolean webkit_dom_html_marquee_element_add_event_listener(WebKitDOMEventTarget* target, const char* eventName, GClosure* handler, gboolean useCapture)
{
    WebCore::HTMLMarqueeElement* coreTarget = static_cast<WebCore::HTMLMarqueeElement*>(WEBKIT_DOM_OBJECT(target)->coreObject);
    return WebKit::GObjectEventListener::addEventListener(G_OBJECT(target), coreTarget, eventName, handler, useCapture);
}

static gboolean webkit_dom_html_marquee_element_remove_event_listener(WebKitDOMEventTarget* target, const char* eventName, GClosure* handler, gboolean useCapture)
{
    WebCore::HTMLMarqueeElement* coreTarget = static_cast<WebCore::HTMLMarqueeElement*>(WEBKIT_DOM_OBJECT(target)->coreObject);
    return WebKit::GObjectEventListener::removeEventListener(G_OBJECT(target), coreTarget, eventName, handler, useCapture);
}

static void webkit_dom_html_marquee_element_dom_event_target_init(WebKitDOMEventTargetIface* iface)
{
    iface->dispatch_event = webkit_dom_html_marquee_element_dispatch_event;
    iface->add_event_listener = webkit_dom_html_marquee_element_add_event_listener;
    iface->remove_event_listener = webkit_dom_html_marquee_element_remove_event_listener;
}

G_DEFINE_TYPE_WITH_CODE(WebKitDOMHTMLMarqueeElement, webkit_dom_html_marquee_element, WEBKIT_DOM_TYPE_HTML_ELEMENT, G_IMPLEMENT_INTERFACE(WEBKIT_DOM_TYPE_EVENT_TARGET, webkit_dom_html_marquee_element_dom_event_target_init))

static void webkit_dom_html_marquee_element_class_init(WebKitDOMHTMLMarqueeElementClass* requestClass)
{
}

static void webkit_dom_html_marquee_element_init(WebKitDOMHTMLMarqueeElement* request)
{
    UNUSED_PARAM(request);
}

void webkit_dom_html_marquee_element_start(WebKitDOMHTMLMarqueeElement* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_if_fail(WEBKIT_DOM_IS_HTML_MARQUEE_ELEMENT(self));
    WebCore::HTMLMarqueeElement* item = WebKit::core(self);
    item->start();
}

void webkit_dom_html_marquee_element_stop(WebKitDOMHTMLMarqueeElement* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_if_fail(WEBKIT_DOM_IS_HTML_MARQUEE_ELEMENT(self));
    WebCore::HTMLMarqueeElement* item = WebKit::core(self);
    item->stop();
}
G_GNUC_END_IGNORE_DEPRECATIONS;
