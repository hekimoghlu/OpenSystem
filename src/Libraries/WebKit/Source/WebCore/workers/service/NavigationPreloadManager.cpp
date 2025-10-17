/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
#include "NavigationPreloadManager.h"

#include "ServiceWorkerContainer.h"
#include "ServiceWorkerRegistration.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigationPreloadManager);

void NavigationPreloadManager::enable(Promise&& promise)
{
    m_registration.container().enableNavigationPreload(m_registration.identifier(), WTFMove(promise));
}

void NavigationPreloadManager::disable(Promise&& promise)
{
    m_registration.container().disableNavigationPreload(m_registration.identifier(), WTFMove(promise));
}

void NavigationPreloadManager::setHeaderValue(String&& value, Promise&& promise)
{
    m_registration.container().setNavigationPreloadHeaderValue(m_registration.identifier(), WTFMove(value), WTFMove(promise));
}

void NavigationPreloadManager::getState(StatePromise&& promise)
{
    m_registration.container().getNavigationPreloadState(m_registration.identifier(), WTFMove(promise));
}

} // namespace WebCore
