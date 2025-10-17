/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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
#include "CookieStoreManager.h"

#include "CookieStoreGetOptions.h"
#include "JSDOMPromiseDeferred.h"
#include "ServiceWorkerRegistration.h"
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

Ref<CookieStoreManager> CookieStoreManager::create(ServiceWorkerRegistration& serviceWorkerRegistration)
{
    return adoptRef(*new CookieStoreManager(serviceWorkerRegistration));
}

CookieStoreManager::CookieStoreManager(ServiceWorkerRegistration& serviceWorkerRegistration)
    : m_serviceWorkerRegistration(serviceWorkerRegistration)
{
}

CookieStoreManager::~CookieStoreManager() = default;

void CookieStoreManager::subscribe(Vector<CookieStoreGetOptions>&& subscriptions, Ref<DeferredPromise>&& promise)
{
    if (RefPtr registration = m_serviceWorkerRegistration.get()) {
        registration->addCookieChangeSubscriptions(WTFMove(subscriptions), WTFMove(promise));
        return;
    }

    promise->reject(Exception { ExceptionCode::InvalidStateError, "There is no service worker registration"_s });
}

void CookieStoreManager::unsubscribe(Vector<CookieStoreGetOptions>&& subscriptions, Ref<DeferredPromise>&& promise)
{
    if (RefPtr registration = m_serviceWorkerRegistration.get()) {
        registration->removeCookieChangeSubscriptions(WTFMove(subscriptions), WTFMove(promise));
        return;
    }

    promise->reject(Exception { ExceptionCode::InvalidStateError, "There is no service worker registration"_s });
}

void CookieStoreManager::getSubscriptions(Ref<DeferredPromise>&& promise)
{
    if (RefPtr registration = m_serviceWorkerRegistration.get()) {
        registration->cookieChangeSubscriptions(WTFMove(promise));
        return;
    }

    promise->reject(Exception { ExceptionCode::InvalidStateError, "There is no service worker registration"_s });
}

} // namespace WebCore
