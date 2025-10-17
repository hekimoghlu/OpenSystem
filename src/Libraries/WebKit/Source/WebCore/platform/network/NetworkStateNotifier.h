/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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

#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>
#include "Timer.h"

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS WebNetworkStateObserver;
#endif

#if PLATFORM(MAC)
typedef const struct __SCDynamicStore * SCDynamicStoreRef;
#endif

#if PLATFORM(WIN)
#include <windows.h>
#endif

namespace WebCore {

class NetworkStateNotifier {
    WTF_MAKE_NONCOPYABLE(NetworkStateNotifier);

public:
    WEBCORE_EXPORT static NetworkStateNotifier& singleton();

    WEBCORE_EXPORT bool onLine();
    WEBCORE_EXPORT void addListener(Function<void(bool isOnLine)>&&);

private:
    friend NeverDestroyed<NetworkStateNotifier>;

    NetworkStateNotifier();

    void updateStateWithoutNotifying();
    void updateState();
    void updateStateSoon();
    void startObserving();

#if PLATFORM(WIN)
    void registerForAddressChange();
    static void CALLBACK addressChangeCallback(void*, BOOLEAN timedOut);
#endif

#if USE(GLIB)
    static void networkChangedCallback(NetworkStateNotifier*);
#endif

    std::optional<bool> m_isOnLine;
    Vector<Function<void(bool)>> m_listeners;
    Timer m_updateStateTimer;

#if PLATFORM(IOS_FAMILY)
    RetainPtr<WebNetworkStateObserver> m_observer;
#endif

#if PLATFORM(MAC)
    RetainPtr<SCDynamicStoreRef> m_store;
#endif

#if PLATFORM(WIN)
    HANDLE m_waitHandle;
    OVERLAPPED m_overlapped;
#endif
};

} // namespace WebCore
