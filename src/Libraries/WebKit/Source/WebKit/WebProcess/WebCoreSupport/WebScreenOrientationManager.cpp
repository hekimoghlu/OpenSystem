/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
#include "WebScreenOrientationManager.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebProcess.h"
#include "WebScreenOrientationManagerMessages.h"
#include "WebScreenOrientationManagerProxyMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebScreenOrientationManager);

WebScreenOrientationManager::WebScreenOrientationManager(WebPage& page)
    : m_page(page)
{
    WebProcess::singleton().addMessageReceiver(Messages::WebScreenOrientationManager::messageReceiverName(), m_page->identifier(), *this);
}

WebScreenOrientationManager::~WebScreenOrientationManager()
{
    WebProcess::singleton().removeMessageReceiver(Messages::WebScreenOrientationManager::messageReceiverName(), m_page->identifier());
}

void WebScreenOrientationManager::ref() const
{
    m_page->ref();
}

void WebScreenOrientationManager::deref() const
{
    m_page->deref();
}

Ref<WebPage> WebScreenOrientationManager::protectedPage() const
{
    return m_page.get();
}

WebCore::ScreenOrientationType WebScreenOrientationManager::currentOrientation()
{
    if (m_currentOrientation)
        return *m_currentOrientation;

    auto sendResult = protectedPage()->sendSync(Messages::WebScreenOrientationManagerProxy::CurrentOrientation { });
    auto [currentOrientation] = sendResult.takeReplyOr(WebCore::naturalScreenOrientationType());
    if (!m_observers.isEmptyIgnoringNullReferences())
        m_currentOrientation = currentOrientation;
    return currentOrientation;
}

void WebScreenOrientationManager::orientationDidChange(WebCore::ScreenOrientationType orientation)
{
    m_currentOrientation = orientation;
    for (auto& observer : m_observers)
        observer.screenOrientationDidChange(orientation);
}

void WebScreenOrientationManager::lock(WebCore::ScreenOrientationLockType lockType, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& completionHandler)
{
    protectedPage()->sendWithAsyncReply(Messages::WebScreenOrientationManagerProxy::Lock { lockType }, WTFMove(completionHandler));
}

void WebScreenOrientationManager::unlock()
{
    protectedPage()->send(Messages::WebScreenOrientationManagerProxy::Unlock { });
}

void WebScreenOrientationManager::addObserver(WebCore::ScreenOrientationManagerObserver& observer)
{
    bool wasEmpty = m_observers.isEmptyIgnoringNullReferences();
    m_observers.add(observer);
    if (wasEmpty)
        protectedPage()->send(Messages::WebScreenOrientationManagerProxy::SetShouldSendChangeNotification { true });
}

void WebScreenOrientationManager::removeObserver(WebCore::ScreenOrientationManagerObserver& observer)
{
    m_observers.remove(observer);
    if (m_observers.isEmptyIgnoringNullReferences()) {
        m_currentOrientation = std::nullopt;
        protectedPage()->send(Messages::WebScreenOrientationManagerProxy::SetShouldSendChangeNotification { false });
    }
}

} // namespace WebKit
