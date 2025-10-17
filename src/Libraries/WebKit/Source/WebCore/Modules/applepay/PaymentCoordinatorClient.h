/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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

#if ENABLE(APPLE_PAY)

#include "ApplePaySessionPaymentRequest.h"
#include "ApplePaySetupFeatureWebCore.h"
#include <wtf/AbstractRefCounted.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>

namespace WebCore {

class Document;
class PaymentCoordinator;
class PaymentMerchantSession;
struct ApplePayCouponCodeUpdate;
struct ApplePayPaymentAuthorizationResult;
struct ApplePayPaymentMethodUpdate;
struct ApplePaySetupConfiguration;
struct ApplePayShippingContactUpdate;
struct ApplePayShippingMethodUpdate;

class PaymentCoordinatorClient : public AbstractRefCounted {
public:
    bool supportsVersion(unsigned version) const;

    virtual std::optional<String> validatedPaymentNetwork(const String&) const = 0;
    virtual bool canMakePayments() = 0;
    virtual void canMakePaymentsWithActiveCard(const String& merchantIdentifier, const String& domainName, CompletionHandler<void(bool)>&&) = 0;
    virtual void openPaymentSetup(const String& merchantIdentifier, const String& domainName, CompletionHandler<void(bool)>&&) = 0;

    virtual bool showPaymentUI(const URL& originatingURL, const Vector<URL>& linkIconURLs, const ApplePaySessionPaymentRequest&) = 0;
    virtual void completeMerchantValidation(const PaymentMerchantSession&) = 0;
    virtual void completeShippingMethodSelection(std::optional<ApplePayShippingMethodUpdate>&&) = 0;
    virtual void completeShippingContactSelection(std::optional<ApplePayShippingContactUpdate>&&) = 0;
    virtual void completePaymentMethodSelection(std::optional<ApplePayPaymentMethodUpdate>&&) = 0;
#if ENABLE(APPLE_PAY_COUPON_CODE)
    virtual void completeCouponCodeChange(std::optional<ApplePayCouponCodeUpdate>&&) = 0;
#endif
    virtual void completePaymentSession(ApplePayPaymentAuthorizationResult&&) = 0;
    virtual void abortPaymentSession() = 0;
    virtual void cancelPaymentSession() = 0;

    virtual bool isMockPaymentCoordinator() const { return false; }
    virtual bool isWebPaymentCoordinator() const { return false; }

    virtual void getSetupFeatures(const ApplePaySetupConfiguration&, const URL&, CompletionHandler<void(Vector<Ref<ApplePaySetupFeature>>&&)>&& completionHandler) { completionHandler({ }); }
    virtual void beginApplePaySetup(const ApplePaySetupConfiguration&, const URL&, Vector<Ref<ApplePaySetupFeature>>&&, CompletionHandler<void(bool)>&& completionHandler) { completionHandler(false); }
    virtual void endApplePaySetup() { }

    virtual ~PaymentCoordinatorClient() = default;
};

}

#endif
