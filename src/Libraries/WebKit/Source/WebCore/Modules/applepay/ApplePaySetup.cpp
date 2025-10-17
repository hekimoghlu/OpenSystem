/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#include "ApplePaySetupWebCore.h"

#if ENABLE(APPLE_PAY)

#include "Document.h"
#include "JSApplePaySetupFeature.h"
#include "JSDOMPromiseDeferred.h"
#include "Page.h"
#include "PaymentCoordinator.h"
#include "PaymentCoordinatorClient.h"
#include "PaymentSession.h"
#include "Settings.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {

static bool shouldDiscloseFeatures(Document& document)
{
    auto* page = document.page();
    if (!page || page->usesEphemeralSession())
        return false;

    return document.settings().applePayCapabilityDisclosureAllowed();
}

void ApplePaySetup::getSetupFeatures(Document& document, SetupFeaturesPromise&& promise)
{
    auto canCall = PaymentSession::canCreateSession(document);
    if (canCall.hasException()) {
        promise.reject(canCall.releaseException());
        return;
    }

    auto page = document.page();
    if (!page) {
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    if (m_setupFeaturesPromise) {
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    // Resolve with an empty sequence of features if Apple Pay capability disclosure is not allowed.
    if (!shouldDiscloseFeatures(document)) {
        promise.resolve({ });
        return;
    }

    m_setupFeaturesPromise = WTFMove(promise);

    page->paymentCoordinator().getSetupFeatures(m_configuration, document.url(), [this, pendingActivity = makePendingActivity(*this)](Vector<Ref<ApplePaySetupFeature>>&& setupFeatures) {
        if (m_setupFeaturesPromise)
            std::exchange(m_setupFeaturesPromise, std::nullopt)->resolve(WTFMove(setupFeatures));
    });
}

void ApplePaySetup::begin(Document& document, Vector<Ref<ApplePaySetupFeature>>&& features, BeginPromise&& promise)
{
    auto canCall = PaymentSession::canCreateSession(document);
    if (canCall.hasException()) {
        promise.reject(canCall.releaseException());
        return;
    }

    if (!UserGestureIndicator::processingUserGesture()) {
        promise.reject(Exception { ExceptionCode::InvalidAccessError, "Must call ApplePaySetup.begin from a user gesture handler."_s });
        return;
    }

    RefPtr page = document.page();
    if (!page) {
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    if (m_beginPromise) {
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    m_beginPromise = WTFMove(promise);
    m_pendingActivity = makePendingActivity(*this);

    page->paymentCoordinator().beginApplePaySetup(m_configuration, page->mainFrameURL(), WTFMove(features), [this](bool result) {
        if (m_beginPromise)
            std::exchange(m_beginPromise, std::nullopt)->resolve(result);
    });
}

Ref<ApplePaySetup> ApplePaySetup::create(ScriptExecutionContext& context, ApplePaySetupConfiguration&& configuration)
{
    auto setup = adoptRef(*new ApplePaySetup(context, WTFMove(configuration)));
    setup->suspendIfNeeded();
    return setup;
}

ApplePaySetup::ApplePaySetup(ScriptExecutionContext& context, ApplePaySetupConfiguration&& configuration)
    : ActiveDOMObject(&context)
    , m_configuration(WTFMove(configuration))
{
}

void ApplePaySetup::stop()
{
    if (m_setupFeaturesPromise)
        std::exchange(m_setupFeaturesPromise, std::nullopt)->reject(Exception { ExceptionCode::AbortError });

    if (m_beginPromise)
        std::exchange(m_beginPromise, std::nullopt)->reject(Exception { ExceptionCode::AbortError });

    if (auto page = downcast<Document>(*scriptExecutionContext()).page())
        page->paymentCoordinator().endApplePaySetup();

    m_pendingActivity = nullptr;
}

void ApplePaySetup::suspend(ReasonForSuspension)
{
    stop();
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
