/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
#import "WebExtensionAPIWebRequestEvent.h"

#import "MessageSenderInlines.h"
#import "ResourceLoadInfo.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionTabIdentifier.h"
#import "WebExtensionWindowIdentifier.h"
#import "WebFrame.h"
#import "WebProcess.h"
#import "_WKWebExtensionWebRequestFilter.h"

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

void WebExtensionAPIWebRequestEvent::enumerateListeners(WebExtensionTabIdentifier tabIdentifier, WebExtensionWindowIdentifier windowIdentifier, const ResourceLoadInfo& resourceLoadInfo, const Function<void(WebExtensionCallbackHandler&, const Vector<String>&)>& function)
{
    if (m_listeners.isEmpty())
        return;

    auto resourceType = toWebExtensionWebRequestResourceType(resourceLoadInfo);
    auto resourceURL = resourceLoadInfo.originalURL;

    // Copy the listeners since call() can trigger a mutation of the listeners.
    auto listenersCopy = m_listeners;

    for (auto& listener : listenersCopy) {
        auto* filter = listener.filter.get();
        if (filter && ![filter matchesRequestForResourceOfType:resourceType URL:resourceURL tabID:toWebAPI(tabIdentifier) windowID:toWebAPI(windowIdentifier)])
            continue;

        function(*listener.callback, listener.extraInfo);
    }
}

void WebExtensionAPIWebRequestEvent::invokeListenersWithArgument(NSDictionary *argument, WebExtensionTabIdentifier tabIdentifier, WebExtensionWindowIdentifier windowIdentifier, const ResourceLoadInfo& resourceLoadInfo)
{
    enumerateListeners(tabIdentifier, windowIdentifier, resourceLoadInfo, [argument = RetainPtr { argument }](auto& listener, auto&) {
        listener.call(argument.get());
    });
}

void WebExtensionAPIWebRequestEvent::addListener(WebCore::FrameIdentifier frameIdentifier, RefPtr<WebExtensionCallbackHandler> listener, NSDictionary *filter, NSArray *extraInfoArray, NSString **outExceptionString)
{
    _WKWebExtensionWebRequestFilter *parsedFilter;
    if (filter) {
        parsedFilter = [[_WKWebExtensionWebRequestFilter alloc] initWithDictionary:filter outErrorMessage:outExceptionString];
        if (!parsedFilter)
            return;
    }

    auto extraInfo = makeVector<String>(extraInfoArray);
    extraInfo.removeAllMatching([&](auto& item) {
        return item != "requestBody"_s && item != "requestHeaders"_s && item != "responseHeaders"_s;
    });
    extraInfo.shrinkToFit();

    m_frameIdentifier = frameIdentifier;
    m_listeners.append({ listener, parsedFilter, WTFMove(extraInfo) });

    WebProcess::singleton().send(Messages::WebExtensionContext::AddListener(*m_frameIdentifier, m_type, contentWorldType()), extensionContext().identifier());
}

void WebExtensionAPIWebRequestEvent::removeListener(WebCore::FrameIdentifier frameIdentifier, RefPtr<WebExtensionCallbackHandler> listener)
{
    auto removedCount = m_listeners.removeAllMatching([&](auto& entry) {
        return entry.callback->callbackFunction() == listener->callbackFunction();
    });

    if (!removedCount)
        return;

    ASSERT(frameIdentifier == m_frameIdentifier);

    WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), removedCount), extensionContext().identifier());
}

bool WebExtensionAPIWebRequestEvent::hasListener(RefPtr<WebExtensionCallbackHandler> listener)
{
    return m_listeners.containsIf([&](auto& entry) {
        return entry.callback->callbackFunction() == listener->callbackFunction();
    });
}

void WebExtensionAPIWebRequestEvent::removeAllListeners()
{
    if (m_listeners.isEmpty())
        return;

    WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), m_listeners.size()), extensionContext().identifier());

    m_listeners.clear();
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
