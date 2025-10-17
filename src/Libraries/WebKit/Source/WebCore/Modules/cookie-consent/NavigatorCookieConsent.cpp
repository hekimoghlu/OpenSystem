/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include "NavigatorCookieConsent.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "CookieConsentDecisionResult.h"
#include "ExceptionCode.h"
#include "JSDOMPromiseDeferred.h"
#include "Navigator.h"
#include "Page.h"
#include "RequestCookieConsentOptions.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorCookieConsent);

void NavigatorCookieConsent::requestCookieConsent(Navigator& navigator, RequestCookieConsentOptions&& options, Ref<DeferredPromise>&& promise)
{
    from(navigator).requestCookieConsent(WTFMove(options), WTFMove(promise));
}

void NavigatorCookieConsent::requestCookieConsent(RequestCookieConsentOptions&& options, Ref<DeferredPromise>&& promise)
{
    // FIXME: Support the 'More info' option.
    UNUSED_PARAM(options);

    RefPtr frame = m_navigator->frame();
    if (!frame || !frame->isMainFrame() || !frame->page()) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    frame->page()->chrome().client().requestCookieConsent([promise = WTFMove(promise)] (CookieConsentDecisionResult result) {
        switch (result) {
        case CookieConsentDecisionResult::NotSupported:
            promise->reject(ExceptionCode::NotSupportedError);
            break;
        case CookieConsentDecisionResult::Consent:
            promise->resolve<IDLBoolean>(true);
            break;
        case CookieConsentDecisionResult::Dissent:
            promise->resolve<IDLBoolean>(false);
            break;
        }
    });
}

NavigatorCookieConsent& NavigatorCookieConsent::from(Navigator& navigator)
{
    if (auto supplement = static_cast<NavigatorCookieConsent*>(Supplement<Navigator>::from(&navigator, supplementName())))
        return *supplement;

    auto newSupplement = makeUnique<NavigatorCookieConsent>(navigator);
    auto supplement = newSupplement.get();
    provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    return *supplement;
}

} // namespace WebCore
