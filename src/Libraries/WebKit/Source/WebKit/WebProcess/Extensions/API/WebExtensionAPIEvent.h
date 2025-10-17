/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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

#include "JSWebExtensionAPIEvent.h"
#include "JSWebExtensionWrappable.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionEventListenerType.h"

OBJC_CLASS JSValue;

namespace WebKit {

class WebExtensionAPIEvent : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIEvent, event, event);

public:
    using ListenerVector = Vector<RefPtr<WebExtensionCallbackHandler>>;

#if PLATFORM(COCOA)
    void invokeListeners();
    void invokeListenersWithArgument(id argument);
    void invokeListenersWithArgument(id argument1, id argument2);
    void invokeListenersWithArgument(id argument1, id argument2, id argument3);
#endif

    const ListenerVector& listeners() const { return m_listeners; }

    void addListener(WebCore::FrameIdentifier, RefPtr<WebExtensionCallbackHandler>);
    void removeListener(WebCore::FrameIdentifier, RefPtr<WebExtensionCallbackHandler>);
    bool hasListener(RefPtr<WebExtensionCallbackHandler>);

    void removeAllListeners();

    virtual ~WebExtensionAPIEvent()
    {
        removeAllListeners();
    }

private:
    explicit WebExtensionAPIEvent(const WebExtensionAPIObject& parentObject, WebExtensionEventListenerType type)
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
