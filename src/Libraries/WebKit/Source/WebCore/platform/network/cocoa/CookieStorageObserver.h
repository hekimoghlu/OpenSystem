/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#include <pal/spi/cf/CFNetworkSPI.h>
#include <wtf/Function.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS NSHTTPCookieStorage;
OBJC_CLASS WebCookieObserverAdapter;

namespace WebCore {
class CookieStorageObserver;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::CookieStorageObserver> : std::true_type { };
}

namespace WebCore {

// Use eager initialization for the WeakPtrFactory since we construct WeakPtrs on a non-main thread.
class WEBCORE_EXPORT CookieStorageObserver : public CanMakeWeakPtr<CookieStorageObserver, WeakPtrFactoryInitialization::Eager> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(CookieStorageObserver, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(CookieStorageObserver);
public:
    explicit CookieStorageObserver(NSHTTPCookieStorage *);
    ~CookieStorageObserver();

    void startObserving(Function<void()>&& callback);
    void stopObserving();

    void cookiesDidChange();

private:
    RetainPtr<NSHTTPCookieStorage> m_cookieStorage;
    bool m_hasRegisteredInternalsForNotifications { false };
    RetainPtr<WebCookieObserverAdapter> m_observerAdapter;
    Function<void()> m_cookieChangeCallback;
};

} // namespace WebCore
