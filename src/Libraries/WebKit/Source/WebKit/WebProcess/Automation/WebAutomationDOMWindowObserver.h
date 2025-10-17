/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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

#include <WebCore/LocalDOMWindow.h>
#include <wtf/Forward.h>

namespace WebCore {
class LocalFrame;
}

namespace WebKit {

class WebAutomationDOMWindowObserver final : public RefCounted<WebAutomationDOMWindowObserver>, public WebCore::LocalDOMWindowObserver {
public:
    static Ref<WebAutomationDOMWindowObserver> create(WebCore::LocalDOMWindow& window, WTF::Function<void(WebAutomationDOMWindowObserver&)>&& callback)
    {
        return adoptRef(*new WebAutomationDOMWindowObserver(window, WTFMove(callback)));
    }

    ~WebAutomationDOMWindowObserver();

    // All of these observer callbacks are interpreted as a signal that a frame has been detached and
    // can no longer accept new commands nor finish pending commands (eg, evaluating JavaScript).
    void willDestroyGlobalObjectInCachedFrame() final;
    void willDestroyGlobalObjectInFrame() final;
    void willDetachGlobalObjectFromFrame() final;

private:
    WebAutomationDOMWindowObserver(WebCore::LocalDOMWindow&, WTF::Function<void(WebAutomationDOMWindowObserver&)>&&);

    WeakPtr<WebCore::LocalDOMWindow, WebCore::WeakPtrImplWithEventTargetData> m_window;
    bool m_wasDetached { false };
    WTF::Function<void(WebAutomationDOMWindowObserver&)> m_callback;
};

} // namespace WebKit
