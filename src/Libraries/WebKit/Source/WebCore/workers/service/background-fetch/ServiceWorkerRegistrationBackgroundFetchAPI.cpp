/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
#include "ServiceWorkerRegistrationBackgroundFetchAPI.h"

#include "BackgroundFetchManager.h"
#include "ServiceWorkerRegistration.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ServiceWorkerRegistrationBackgroundFetchAPI);

ServiceWorkerRegistrationBackgroundFetchAPI::ServiceWorkerRegistrationBackgroundFetchAPI(ServiceWorkerRegistration& serviceWorkerRegistration)
    : m_serviceWorkerRegistration(serviceWorkerRegistration)
{
}

ServiceWorkerRegistrationBackgroundFetchAPI::~ServiceWorkerRegistrationBackgroundFetchAPI()
{
}

BackgroundFetchManager& ServiceWorkerRegistrationBackgroundFetchAPI::backgroundFetch(ServiceWorkerRegistration& serviceWorkerRegistration)
{
    return ServiceWorkerRegistrationBackgroundFetchAPI::from(serviceWorkerRegistration).backgroundFetchManager();
}

RefPtr<BackgroundFetchManager> ServiceWorkerRegistrationBackgroundFetchAPI::backgroundFetchIfCreated(ServiceWorkerRegistration& serviceWorkerRegistration)
{
    return ServiceWorkerRegistrationBackgroundFetchAPI::from(serviceWorkerRegistration).m_backgroundFetchManager;
}

BackgroundFetchManager& ServiceWorkerRegistrationBackgroundFetchAPI::backgroundFetchManager()
{
    if (!m_backgroundFetchManager)
        m_backgroundFetchManager = BackgroundFetchManager::create(m_serviceWorkerRegistration);

    return *m_backgroundFetchManager;
}

ServiceWorkerRegistrationBackgroundFetchAPI& ServiceWorkerRegistrationBackgroundFetchAPI::from(ServiceWorkerRegistration& serviceWorkerRegistration)
{
    auto* supplement = static_cast<ServiceWorkerRegistrationBackgroundFetchAPI*>(Supplement<ServiceWorkerRegistration>::from(&serviceWorkerRegistration, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<ServiceWorkerRegistrationBackgroundFetchAPI>(serviceWorkerRegistration);
        supplement = newSupplement.get();
        provideTo(&serviceWorkerRegistration, supplementName(), WTFMove(newSupplement));
    }
    return *supplement;
}

ASCIILiteral ServiceWorkerRegistrationBackgroundFetchAPI::supplementName()
{
    return "ServiceWorkerRegistrationBackgroundFetchAPI"_s;
}

}
