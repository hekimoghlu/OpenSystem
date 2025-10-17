/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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

#import "config.h"
#import "WebExtensionAPIEvent.h"

#import "MessageSenderInlines.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionControllerProxy.h"
#import "WebFrame.h"
#import "WebPage.h"
#import "WebProcess.h"
#import <JavaScriptCore/APICast.h>
#import <JavaScriptCore/ScriptCallStack.h>
#import <JavaScriptCore/ScriptCallStackFactory.h>

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

class JSWebExtensionWrappable;

void WebExtensionAPIEvent::invokeListeners()
{
    if (m_listeners.isEmpty())
        return;

    // Copy the listeners since call() can trigger a mutation of the listeners.
    auto listenersCopy = m_listeners;

    for (RefPtr listener : listenersCopy)
        listener->call();
}

void WebExtensionAPIEvent::invokeListenersWithArgument(id argument1)
{
    if (m_listeners.isEmpty())
        return;

    // Copy the listeners since call() can trigger a mutation of the listeners.
    auto listenersCopy = m_listeners;

    for (RefPtr listener : listenersCopy)
        listener->call(argument1);
}

void WebExtensionAPIEvent::invokeListenersWithArgument(id argument1, id argument2)
{
    if (m_listeners.isEmpty())
        return;

    // Copy the listeners since call() can trigger a mutation of the listeners.
    auto listenersCopy = m_listeners;

    for (RefPtr listener : listenersCopy)
        listener->call(argument1, argument2);
}

void WebExtensionAPIEvent::invokeListenersWithArgument(id argument1, id argument2, id argument3)
{
    if (m_listeners.isEmpty())
        return;

    // Copy the listeners since call() can trigger a mutation of the listeners.
    auto listenersCopy = m_listeners;

    for (RefPtr listener : listenersCopy)
        listener->call(argument1, argument2, argument3);
}

void WebExtensionAPIEvent::addListener(WebCore::FrameIdentifier frameIdentifier, RefPtr<WebExtensionCallbackHandler> listener)
{
    m_frameIdentifier = frameIdentifier;
    m_listeners.append(listener);

    if (!hasExtensionContext()) {
        RefPtr webFrame = WebProcess::singleton().webFrame(m_frameIdentifier);
        RefPtr webPage = webFrame ? webFrame->page() : nullptr;
        RefPtr webExtensionControllerProxy = webPage ? webPage->webExtensionControllerProxy() : nullptr;

        if (webExtensionControllerProxy && webExtensionControllerProxy->inTestingMode()) {
            for (Ref extensionContext : webExtensionControllerProxy->extensionContexts()) {
                extensionContext->addFrameWithExtensionContent(*webFrame);
                WebProcess::singleton().send(Messages::WebExtensionContext::AddListener(*m_frameIdentifier, m_type, contentWorldType()), extensionContext->identifier());
            }
        }

        return;
    }

    WebProcess::singleton().send(Messages::WebExtensionContext::AddListener(*m_frameIdentifier, m_type, contentWorldType()), extensionContext().identifier());
}

void WebExtensionAPIEvent::removeListener(WebCore::FrameIdentifier frameIdentifier, RefPtr<WebExtensionCallbackHandler> listener)
{
    auto removedCount = m_listeners.removeAllMatching([&](auto& entry) {
        return entry->callbackFunction() == listener->callbackFunction();
    });

    if (!removedCount)
        return;

    ASSERT(frameIdentifier == m_frameIdentifier);

    if (!hasExtensionContext()) {
        RefPtr webFrame = WebProcess::singleton().webFrame(m_frameIdentifier);
        RefPtr webPage = webFrame ? webFrame->page() : nullptr;
        RefPtr webExtensionControllerProxy = webPage ? webPage->webExtensionControllerProxy() : nullptr;

        if (webExtensionControllerProxy && webExtensionControllerProxy->inTestingMode()) {
            for (Ref extensionContext : webExtensionControllerProxy->extensionContexts())
                WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), removedCount), extensionContext->identifier());
        }

        return;
    }

    WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), removedCount), extensionContext().identifier());
}

bool WebExtensionAPIEvent::hasListener(RefPtr<WebExtensionCallbackHandler> listener)
{
    return m_listeners.containsIf([&](auto& entry) {
        return entry->callbackFunction() == listener->callbackFunction();
    });
}

void WebExtensionAPIEvent::removeAllListeners()
{
    if (m_listeners.isEmpty())
        return;

    auto removedCount = m_listeners.size();
    m_listeners.clear();

    if (!hasExtensionContext()) {
        RefPtr webFrame = WebProcess::singleton().webFrame(m_frameIdentifier);
        RefPtr webPage = webFrame ? webFrame->page() : nullptr;
        RefPtr webExtensionControllerProxy = webPage ? webPage->webExtensionControllerProxy() : nullptr;

        if (webExtensionControllerProxy && webExtensionControllerProxy->inTestingMode()) {
            for (Ref extensionContext : webExtensionControllerProxy->extensionContexts())
                WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), removedCount), extensionContext->identifier());
        }

        return;
    }

    WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), removedCount), extensionContext().identifier());
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
