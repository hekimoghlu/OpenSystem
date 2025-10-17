/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#include <WebCore/ScreenOrientationManager.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebPage;

class WebScreenOrientationManager final : public WebCore::ScreenOrientationManager, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(WebScreenOrientationManager);
public:
    explicit WebScreenOrientationManager(WebPage&);
    ~WebScreenOrientationManager();

    void ref() const final;
    void deref() const final;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

private:
    void orientationDidChange(WebCore::ScreenOrientationType);

    // ScreenOrientationManager
    WebCore::ScreenOrientationType currentOrientation() final;
    void lock(WebCore::ScreenOrientationLockType, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&) final;
    void unlock() final;
    void addObserver(WebCore::ScreenOrientationManagerObserver&) final;
    void removeObserver(WebCore::ScreenOrientationManagerObserver&) final;

    Ref<WebPage> protectedPage() const;

    WeakRef<WebPage> m_page;
    WeakHashSet<WebCore::ScreenOrientationManagerObserver> m_observers;
    mutable std::optional<WebCore::ScreenOrientationType> m_currentOrientation;
};

} // namespace WebKit
