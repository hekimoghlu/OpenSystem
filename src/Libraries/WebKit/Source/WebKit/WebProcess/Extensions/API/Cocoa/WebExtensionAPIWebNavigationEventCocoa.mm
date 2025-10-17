/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
#import "WebExtensionAPIWebNavigationEvent.h"

#import "MessageSenderInlines.h"
#import "WebExtensionContextMessages.h"
#import "WebProcess.h"
#import "_WKWebExtensionWebNavigationURLFilter.h"

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

void WebExtensionAPIWebNavigationEvent::invokeListenersWithArgument(id argument, NSURL *targetURL)
{
    if (m_listeners.isEmpty())
        return;

    // Copy the listeners since call() can trigger a mutation of the listeners.
    auto listenersCopy = m_listeners;

    for (auto& listener : listenersCopy) {
        auto *filter = listener.second.get();
        if (filter && ![filter matchesURL:targetURL])
            continue;

        listener.first->call(argument);
    }
}

void WebExtensionAPIWebNavigationEvent::addListener(WebCore::FrameIdentifier frameIdentifier, RefPtr<WebExtensionCallbackHandler> listener, NSDictionary *filter, NSString **outExceptionString)
{
    _WKWebExtensionWebNavigationURLFilter *parsedFilter;
    if (filter) {
        parsedFilter = [[_WKWebExtensionWebNavigationURLFilter alloc] initWithDictionary:filter outErrorMessage:outExceptionString];
        if (!parsedFilter)
            return;
    }

    m_frameIdentifier = frameIdentifier;
    m_listeners.append({ listener, parsedFilter });

    WebProcess::singleton().send(Messages::WebExtensionContext::AddListener(*m_frameIdentifier, m_type, contentWorldType()), extensionContext().identifier());
}

void WebExtensionAPIWebNavigationEvent::removeListener(WebCore::FrameIdentifier frameIdentifier, RefPtr<WebExtensionCallbackHandler> listener)
{
    auto removedCount = m_listeners.removeAllMatching([&](auto& entry) {
        return entry.first->callbackFunction() == listener->callbackFunction();
    });

    if (!removedCount)
        return;

    ASSERT(frameIdentifier == m_frameIdentifier);

    WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), removedCount), extensionContext().identifier());
}

bool WebExtensionAPIWebNavigationEvent::hasListener(RefPtr<WebExtensionCallbackHandler> listener)
{
    return m_listeners.containsIf([&](auto& entry) {
        return entry.first->callbackFunction() == listener->callbackFunction();
    });
}

void WebExtensionAPIWebNavigationEvent::removeAllListeners()
{
    if (m_listeners.isEmpty())
        return;

    WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), m_listeners.size()), extensionContext().identifier());

    m_listeners.clear();
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
