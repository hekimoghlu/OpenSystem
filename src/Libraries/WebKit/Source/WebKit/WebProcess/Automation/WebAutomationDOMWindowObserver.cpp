/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#include "WebAutomationDOMWindowObserver.h"

#include <WebCore/Element.h>
#include <WebCore/LocalFrame.h>

namespace WebKit {

WebAutomationDOMWindowObserver::WebAutomationDOMWindowObserver(WebCore::LocalDOMWindow& window, WTF::Function<void(WebAutomationDOMWindowObserver&)>&& callback)
    : m_window(window)
    , m_callback(WTFMove(callback))
{
    ASSERT(m_window->frame());
    m_window->registerObserver(*this);
}

WebAutomationDOMWindowObserver::~WebAutomationDOMWindowObserver()
{
    if (m_window)
        m_window->unregisterObserver(*this);
}

void WebAutomationDOMWindowObserver::willDestroyGlobalObjectInCachedFrame()
{
    Ref<WebAutomationDOMWindowObserver> protectedThis(*this);

    if (!m_wasDetached) {
        ASSERT(m_window && m_window->frame());
        m_callback(*this);
    }

    ASSERT(m_window);
    if (m_window)
        m_window->unregisterObserver(*this);
    m_window = nullptr;
}

void WebAutomationDOMWindowObserver::willDestroyGlobalObjectInFrame()
{
    Ref<WebAutomationDOMWindowObserver> protectedThis(*this);

    if (!m_wasDetached) {
        ASSERT(m_window && m_window->frame());
        m_callback(*this);
    }

    ASSERT(m_window);
    if (m_window)
        m_window->unregisterObserver(*this);
    m_window = nullptr;
}

void WebAutomationDOMWindowObserver::willDetachGlobalObjectFromFrame()
{
    ASSERT(!m_wasDetached);

    Ref<WebAutomationDOMWindowObserver> protectedThis(*this);

    m_wasDetached = true;

    m_callback(*this);

    ASSERT(m_window);
    if (m_window)
        m_window->unregisterObserver(*this);
    m_window = nullptr;
}

} // namespace WebKit
