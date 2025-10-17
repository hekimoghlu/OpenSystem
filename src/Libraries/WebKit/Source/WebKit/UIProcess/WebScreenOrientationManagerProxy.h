/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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

#include "MessageReceiver.h"
#include <WebCore/ScreenOrientationLockType.h>
#include <WebCore/ScreenOrientationType.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class Exception;
}

namespace WebKit {

class WebPageProxy;
struct SharedPreferencesForWebProcess;

class WebScreenOrientationManagerProxy final : public IPC::MessageReceiver, public RefCounted<WebScreenOrientationManagerProxy> {
    WTF_MAKE_TZONE_ALLOCATED(WebScreenOrientationManagerProxy);
public:
    static Ref<WebScreenOrientationManagerProxy> create(WebPageProxy&, WebCore::ScreenOrientationType);
    ~WebScreenOrientationManagerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    void unlockIfNecessary();

    void setCurrentOrientation(WebCore::ScreenOrientationType);

private:
    WebScreenOrientationManagerProxy(WebPageProxy&, WebCore::ScreenOrientationType);

    std::optional<WebCore::Exception> platformShouldRejectLockRequest() const;

    // IPC message handlers.
    void currentOrientation(CompletionHandler<void(WebCore::ScreenOrientationType)>&&);
    void lock(WebCore::ScreenOrientationLockType, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&);
    void unlock();
    void setShouldSendChangeNotification(bool);

    Ref<WebPageProxy> protectedPage() const;

    WeakRef<WebPageProxy> m_page;
    WebCore::ScreenOrientationType m_currentOrientation;
    std::optional<WebCore::ScreenOrientationType> m_currentlyLockedOrientation;
    CompletionHandler<void(std::optional<WebCore::Exception>&&)> m_currentLockRequest;
    bool m_shouldSendChangeNotifications { false };
};

} // namespace WebKit
