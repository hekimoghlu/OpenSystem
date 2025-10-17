/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#include "config.h"
#include "WebExtensionControllerProxy.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include "JSWebExtensionAPINamespace.h"
#include "JSWebExtensionAPIWebPageNamespace.h"
#include "JSWebExtensionWrapper.h"
#include "MessageSenderInlines.h"
#include "WebExtensionAPINamespace.h"
#include "WebExtensionAPIWebPageNamespace.h"
#include "WebExtensionContextProxy.h"
#include "WebExtensionControllerMessages.h"
#include "WebExtensionFrameIdentifier.h"
#include "WebFrame.h"
#include "WebPage.h"
#include "WebProcess.h"

namespace WebKit {

using namespace WebCore;

void WebExtensionControllerProxy::globalObjectIsAvailableForFrame(WebPage& page, WebFrame& frame, DOMWrapperWorld& world)
{
    RefPtr extension = extensionContext(frame, world);
    bool isMainWorld = world.isNormal();

    if (!extension && isMainWorld) {
        addBindingsToWebPageFrameIfNecessary(frame, world);
        return;
    }

    if (!extension)
        return;

    auto context = frame.jsContextForWorld(world);
    auto globalObject = JSContextGetGlobalObject(context);

    auto namespaceObject = JSObjectGetProperty(context, globalObject, toJSString("browser").get(), nullptr);
    if (namespaceObject && JSValueIsObject(context, namespaceObject))
        return;

    extension->addFrameWithExtensionContent(frame);

    if (!isMainWorld)
        extension->setContentScriptWorld(&world);

    auto contentWorldType = isMainWorld ? WebExtensionContentWorldType::Main : WebExtensionContentWorldType::ContentScript;

#if ENABLE(INSPECTOR_EXTENSIONS)
    if (page.isInspectorPage() || extension->isInspectorBackgroundPage(page)) {
        // Inspector pages have a limited set of APIs (like content scripts).
        contentWorldType = WebExtensionContentWorldType::Inspector;
    }
#endif

    namespaceObject = toJS(context, WebExtensionAPINamespace::create(contentWorldType, *extension).ptr());

    JSObjectSetProperty(context, globalObject, toJSString("browser").get(), namespaceObject, kJSPropertyAttributeNone, nullptr);
    JSObjectSetProperty(context, globalObject, toJSString("chrome").get(), namespaceObject, kJSPropertyAttributeNone, nullptr);
}

void WebExtensionControllerProxy::serviceWorkerGlobalObjectIsAvailableForFrame(WebPage& page, WebFrame& frame, DOMWrapperWorld& world)
{
    RELEASE_ASSERT(world.isNormal());

    RefPtr extension = extensionContext(frame, world);
    if (!extension)
        return;

    auto context = frame.jsContextForServiceWorkerWorld(world);
    auto globalObject = JSContextGetGlobalObject(context);

    auto namespaceObject = JSObjectGetProperty(context, globalObject, toJSString("browser").get(), nullptr);
    if (namespaceObject && JSValueIsObject(context, namespaceObject))
        return;

    extension->addFrameWithExtensionContent(frame);

    namespaceObject = toJS(context, WebExtensionAPINamespace::create(WebExtensionContentWorldType::Main, *extension).ptr());

    JSObjectSetProperty(context, globalObject, toJSString("browser").get(), namespaceObject, kJSPropertyAttributeNone, nullptr);
    JSObjectSetProperty(context, globalObject, toJSString("chrome").get(), namespaceObject, kJSPropertyAttributeNone, nullptr);
}

void WebExtensionControllerProxy::addBindingsToWebPageFrameIfNecessary(WebFrame& frame, DOMWrapperWorld& world)
{
    auto context = frame.jsContextForWorld(world);
    auto globalObject = JSContextGetGlobalObject(context);

    auto namespaceObject = JSObjectGetProperty(context, globalObject, toJSString("browser").get(), nullptr);
    if (namespaceObject && JSValueIsObject(context, namespaceObject))
        return;

    namespaceObject = toJS(context, WebExtensionAPIWebPageNamespace::create(WebExtensionContentWorldType::WebPage).ptr());

    JSObjectSetProperty(context, globalObject, toJSString("browser").get(), namespaceObject, kJSPropertyAttributeNone, nullptr);
}

static WebExtensionFrameParameters toFrameParameters(WebFrame& frame, const URL& url, bool includeDocumentIdentifier = true)
{
    auto parentFrameIdentifier = WebExtensionFrameConstants::NoneIdentifier;
    if (RefPtr parentFrame = frame.parentFrame())
        parentFrameIdentifier = toWebExtensionFrameIdentifier(*parentFrame);

    return {
        .url = url,
        .parentFrameIdentifier = parentFrameIdentifier,
        .frameIdentifier = toWebExtensionFrameIdentifier(frame),
        .documentIdentifier = includeDocumentIdentifier ? toDocumentIdentifier(frame) : std::nullopt
    };
}

void WebExtensionControllerProxy::didStartProvisionalLoadForFrame(WebPage& page, WebFrame& frame, const URL& url)
{
    if (!hasLoadedContexts())
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::DidStartProvisionalLoadForFrame(page.webPageProxyIdentifier(), toFrameParameters(frame, url, false), WallTime::now()), identifier());
}

void WebExtensionControllerProxy::didCommitLoadForFrame(WebPage& page, WebFrame& frame, const URL& url)
{
    if (!hasLoadedContexts())
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::DidCommitLoadForFrame(page.webPageProxyIdentifier(), toFrameParameters(frame, url), WallTime::now()), identifier());
}

void WebExtensionControllerProxy::didFinishLoadForFrame(WebPage& page, WebFrame& frame, const URL& url)
{
    if (!hasLoadedContexts())
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::DidFinishLoadForFrame(page.webPageProxyIdentifier(), toFrameParameters(frame, url), WallTime::now()), identifier());
}

void WebExtensionControllerProxy::didFailLoadForFrame(WebPage& page, WebFrame& frame, const URL& url)
{
    if (!hasLoadedContexts())
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::DidFailLoadForFrame(page.webPageProxyIdentifier(), toFrameParameters(frame, url), WallTime::now()), identifier());
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
