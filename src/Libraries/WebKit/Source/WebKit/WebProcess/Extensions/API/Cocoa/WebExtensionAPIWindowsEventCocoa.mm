/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#import "WebExtensionAPIWindowsEvent.h"

#import "CocoaHelpers.h"
#import "MessageSenderInlines.h"
#import "WebExtensionAPIWindows.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionUtilities.h"
#import "WebFrame.h"
#import "WebProcess.h"

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

void WebExtensionAPIWindowsEvent::invokeListenersWithArgument(id argument, OptionSet<WindowTypeFilter> windowTypeFilter)
{
    if (m_listeners.isEmpty())
        return;

    // Copy the listeners since call() can trigger a mutation of the listeners.
    auto listenersCopy = m_listeners;

    for (auto& listener : listenersCopy) {
        if (!listener.second.containsAny(windowTypeFilter))
            continue;

        listener.first->call(argument);
    }
}

void WebExtensionAPIWindowsEvent::addListener(WebCore::FrameIdentifier frameIdentifier, RefPtr<WebExtensionCallbackHandler> listener, NSDictionary *filter, NSString **outExceptionString)
{
    OptionSet<WindowTypeFilter> windowTypeFilter;
    if (!WebExtensionAPIWindows::parseWindowTypesFilter(filter, windowTypeFilter, @"filters", outExceptionString))
        return;

    m_frameIdentifier = frameIdentifier;
    m_listeners.append({ listener, windowTypeFilter });

    WebProcess::singleton().send(Messages::WebExtensionContext::AddListener(*m_frameIdentifier, m_type, contentWorldType()), extensionContext().identifier());
}

void WebExtensionAPIWindowsEvent::removeListener(WebCore::FrameIdentifier frameIdentifier, RefPtr<WebExtensionCallbackHandler> listener)
{
    auto removedCount = m_listeners.removeAllMatching([&](auto& entry) {
        return entry.first->callbackFunction() == listener->callbackFunction();
    });

    if (!removedCount)
        return;

    ASSERT(frameIdentifier == m_frameIdentifier);

    WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), removedCount), extensionContext().identifier());
}

bool WebExtensionAPIWindowsEvent::hasListener(RefPtr<WebExtensionCallbackHandler> listener)
{
    return m_listeners.containsIf([&](auto& entry) {
        return entry.first->callbackFunction() == listener->callbackFunction();
    });
}

void WebExtensionAPIWindowsEvent::removeAllListeners()
{
    if (m_listeners.isEmpty())
        return;

    WebProcess::singleton().send(Messages::WebExtensionContext::RemoveListener(*m_frameIdentifier, m_type, contentWorldType(), m_listeners.size()), extensionContext().identifier());

    m_listeners.clear();
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
