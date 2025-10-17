/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include "WebKitDOMComment.h"

#include <WebCore/CSSImportRule.h>
#include "DOMObjectCache.h"
#include <WebCore/DOMException.h>
#include <WebCore/Document.h>
#include "GObjectEventListener.h"
#include <WebCore/JSExecState.h>
#include "WebKitDOMCommentPrivate.h"
#include "WebKitDOMEventPrivate.h"
#include "WebKitDOMEventTarget.h"
#include "WebKitDOMNodePrivate.h"
#include "WebKitDOMPrivate.h"
#include "ConvertToUTF8String.h"
#include <wtf/GetPtr.h>
#include <wtf/RefPtr.h>

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

WebKitDOMComment* kit(WebCore::Comment* obj)
{
    return WEBKIT_DOM_COMMENT(kit(static_cast<WebCore::Node*>(obj)));
}

WebCore::Comment* core(WebKitDOMComment* request)
{
    return request ? static_cast<WebCore::Comment*>(WEBKIT_DOM_OBJECT(request)->coreObject) : 0;
}

WebKitDOMComment* wrapComment(WebCore::Comment* coreObject)
{
    ASSERT(coreObject);
    return WEBKIT_DOM_COMMENT(g_object_new(WEBKIT_DOM_TYPE_COMMENT, "core-object", coreObject, nullptr));
}

} // namespace WebKit

static gboolean webkit_dom_comment_dispatch_event(WebKitDOMEventTarget* target, WebKitDOMEvent* event, GError** error)
{
    WebCore::Event* coreEvent = WebKit::core(event);
    if (!coreEvent)
        return false;
    WebCore::Comment* coreTarget = static_cast<WebCore::Comment*>(WEBKIT_DOM_OBJECT(target)->coreObject);

    auto result = coreTarget->dispatchEventForBindings(*coreEvent);
    if (result.hasException()) {
        auto description = WebCore::DOMException::description(result.releaseException().code());
        g_set_error_literal(error, g_quark_from_string("WEBKIT_DOM"), description.legacyCode, description.name);
        return false;
    }
    return result.releaseReturnValue();
}

static gboolean webkit_dom_comment_add_event_listener(WebKitDOMEventTarget* target, const char* eventName, GClosure* handler, gboolean useCapture)
{
    WebCore::Comment* coreTarget = static_cast<WebCore::Comment*>(WEBKIT_DOM_OBJECT(target)->coreObject);
    return WebKit::GObjectEventListener::addEventListener(G_OBJECT(target), coreTarget, eventName, handler, useCapture);
}

static gboolean webkit_dom_comment_remove_event_listener(WebKitDOMEventTarget* target, const char* eventName, GClosure* handler, gboolean useCapture)
{
    WebCore::Comment* coreTarget = static_cast<WebCore::Comment*>(WEBKIT_DOM_OBJECT(target)->coreObject);
    return WebKit::GObjectEventListener::removeEventListener(G_OBJECT(target), coreTarget, eventName, handler, useCapture);
}

static void webkit_dom_comment_dom_event_target_init(WebKitDOMEventTargetIface* iface)
{
    iface->dispatch_event = webkit_dom_comment_dispatch_event;
    iface->add_event_listener = webkit_dom_comment_add_event_listener;
    iface->remove_event_listener = webkit_dom_comment_remove_event_listener;
}

G_DEFINE_TYPE_WITH_CODE(WebKitDOMComment, webkit_dom_comment, WEBKIT_DOM_TYPE_CHARACTER_DATA, G_IMPLEMENT_INTERFACE(WEBKIT_DOM_TYPE_EVENT_TARGET, webkit_dom_comment_dom_event_target_init))

static void webkit_dom_comment_class_init(WebKitDOMCommentClass* requestClass)
{
    UNUSED_PARAM(requestClass);
}

static void webkit_dom_comment_init(WebKitDOMComment* request)
{
    UNUSED_PARAM(request);
}

G_GNUC_END_IGNORE_DEPRECATIONS;
