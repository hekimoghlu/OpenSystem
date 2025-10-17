/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#include "NavigatorLoginStatus.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "DocumentInlines.h"
#include "JSDOMPromiseDeferred.h"
#include "Navigator.h"
#include "Page.h"
#include "RegistrableDomain.h"
#include "SecurityOrigin.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorLoginStatus);

NavigatorLoginStatus* NavigatorLoginStatus::from(Navigator& navigator)
{
    auto* supplement = static_cast<NavigatorLoginStatus*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorLoginStatus>(navigator);
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

ASCIILiteral NavigatorLoginStatus::supplementName()
{
    return "NavigatorLoginStatus"_s;
}

void NavigatorLoginStatus::setStatus(Navigator& navigator, IsLoggedIn isLoggedIn, Ref<DeferredPromise>&& promise)
{
    NavigatorLoginStatus::from(navigator)->setStatus(isLoggedIn, WTFMove(promise));
}

void NavigatorLoginStatus::isLoggedIn(Navigator& navigator, Ref<DeferredPromise>&& promise)
{
    NavigatorLoginStatus::from(navigator)->isLoggedIn(WTFMove(promise));
}

bool NavigatorLoginStatus::hasSameOrigin() const
{
    RefPtr document = m_navigator.document();
    if (!document)
        return false;
    Ref origin = document->securityOrigin();
    bool isSameSite = true;
    for (RefPtr parentDocument = document->parentDocument(); parentDocument; parentDocument = parentDocument->parentDocument()) {
        if (!origin->isSameOriginAs(parentDocument->protectedSecurityOrigin())) {
            isSameSite = false;
            break;
        }
    }
    return isSameSite;
}

void NavigatorLoginStatus::setStatus(IsLoggedIn isLoggedIn, Ref<DeferredPromise>&& promise)
{
    RefPtr document = m_navigator.document();
    if (!document || !hasSameOrigin()) {
        promise->reject();
        return;
    }

    RefPtr page = document->page();
    if (!page) {
        promise->reject();
        return;
    }
    page->chrome().client().setLoginStatus(RegistrableDomain::uncheckedCreateFromHost(document->securityOrigin().host()), isLoggedIn, [promise = WTFMove(promise)] {
        promise->resolve();
    });
}

void NavigatorLoginStatus::isLoggedIn(Ref<DeferredPromise>&& promise)
{
    RefPtr document = m_navigator.document();
    if (!document) {
        promise->reject();
        return;
    }

    RefPtr page = document->page();
    if (!page) {
        promise->reject();
        return;
    }
    page->chrome().client().isLoggedIn(RegistrableDomain::uncheckedCreateFromHost(document->securityOrigin().host()), [promise = WTFMove(promise)] (bool isLoggedIn) {
        promise->resolve<IDLBoolean>(isLoggedIn);
    });
}

} // namespace WebCore
