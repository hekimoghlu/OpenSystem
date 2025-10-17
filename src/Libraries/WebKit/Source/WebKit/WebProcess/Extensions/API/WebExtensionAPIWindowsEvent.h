/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#pragma once

#if ENABLE(WK_WEB_EXTENSIONS)

#include "JSWebExtensionAPIWindowsEvent.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionWindow.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace WebKit {

class WebExtensionAPIWindowsEvent : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIWindowsEvent, windowsEvent, event);

public:
    using WindowTypeFilter = WebExtensionWindow::TypeFilter;
    using FilterAndCallbackPair = std::pair<RefPtr<WebExtensionCallbackHandler>, OptionSet<WindowTypeFilter>>;
    using ListenerVector = Vector<FilterAndCallbackPair>;

#if PLATFORM(COCOA)
    void invokeListenersWithArgument(id argument, OptionSet<WindowTypeFilter>);
#endif

    const ListenerVector& listeners() const { return m_listeners; }

    void addListener(WebCore::FrameIdentifier, RefPtr<WebExtensionCallbackHandler>, NSDictionary *filter, NSString **outExceptionString);
    void removeListener(WebCore::FrameIdentifier, RefPtr<WebExtensionCallbackHandler>);
    bool hasListener(RefPtr<WebExtensionCallbackHandler>);

    void removeAllListeners();

    virtual ~WebExtensionAPIWindowsEvent()
    {
        removeAllListeners();
    }

private:
    explicit WebExtensionAPIWindowsEvent(const WebExtensionAPIObject& parentObject, WebExtensionEventListenerType type)
        : WebExtensionAPIObject(parentObject)
        , m_type(type)
    {
        setPropertyPath(toAPIString(type), &parentObject);
    }

    Markable<WebCore::FrameIdentifier> m_frameIdentifier;
    WebExtensionEventListenerType m_type;
    ListenerVector m_listeners;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
