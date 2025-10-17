/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

#if USE(PASSKIT) && ENABLE(APPLE_PAY)

#include "CocoaWindow.h"
#include <WebCore/ApplePaySessionPaymentRequest.h>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS UIViewController;
OBJC_CLASS WKPaymentAuthorizationDelegate;

namespace WebCore {
class Payment;
class PaymentContact;
class PaymentMerchantSession;
class PaymentMethod;
class PaymentSessionError;
struct ApplePayCouponCodeUpdate;
struct ApplePayPaymentAuthorizationResult;
struct ApplePayPaymentMethodUpdate;
struct ApplePayShippingContactUpdate;
struct ApplePayShippingMethod;
struct ApplePayShippingMethodUpdate;
}

namespace WebKit {

class PaymentAuthorizationPresenter : public RefCountedAndCanMakeWeakPtr<PaymentAuthorizationPresenter> {
    WTF_MAKE_TZONE_ALLOCATED(PaymentAuthorizationPresenter);
    WTF_MAKE_NONCOPYABLE(PaymentAuthorizationPresenter);
public:
    struct Client : public AbstractRefCountedAndCanMakeWeakPtr<Client> {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        virtual ~Client() = default;

        virtual void presenterDidAuthorizePayment(PaymentAuthorizationPresenter&, const WebCore::Payment&) = 0;
        virtual void presenterDidFinish(PaymentAuthorizationPresenter&, WebCore::PaymentSessionError&&) = 0;
        virtual void presenterDidSelectPaymentMethod(PaymentAuthorizationPresenter&, const WebCore::PaymentMethod&) = 0;
        virtual void presenterDidSelectShippingContact(PaymentAuthorizationPresenter&, const WebCore::PaymentContact&) = 0;
        virtual void presenterDidSelectShippingMethod(PaymentAuthorizationPresenter&, const WebCore::ApplePayShippingMethod&) = 0;
#if HAVE(PASSKIT_COUPON_CODE)
        virtual void presenterDidChangeCouponCode(PaymentAuthorizationPresenter&, const String& couponCode) = 0;
#endif
        virtual void presenterWillValidateMerchant(PaymentAuthorizationPresenter&, const URL&) = 0;
        virtual CocoaWindow *presentingWindowForPaymentAuthorization(PaymentAuthorizationPresenter&) const = 0;
    };

    virtual ~PaymentAuthorizationPresenter() = default;

    RefPtr<Client> protectedClient() { return m_client.get(); }

    void completeMerchantValidation(const WebCore::PaymentMerchantSession&);
    void completePaymentMethodSelection(std::optional<WebCore::ApplePayPaymentMethodUpdate>&&);
    void completePaymentSession(WebCore::ApplePayPaymentAuthorizationResult&&);
    void completeShippingContactSelection(std::optional<WebCore::ApplePayShippingContactUpdate>&&);
    void completeShippingMethodSelection(std::optional<WebCore::ApplePayShippingMethodUpdate>&&);
#if HAVE(PASSKIT_COUPON_CODE)
    void completeCouponCodeChange(std::optional<WebCore::ApplePayCouponCodeUpdate>&&);
#endif

    virtual void dismiss() = 0;
#if PLATFORM(IOS_FAMILY)
    virtual void present(UIViewController *, CompletionHandler<void(bool)>&&) = 0;
#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
    virtual void presentInScene(const String& sceneIdentifier, const String& bundleIdentifier, CompletionHandler<void(bool)>&&) = 0;
    const String& sceneIdentifier() const { return m_sceneIdentifier; }
    const String& bundleIdentifier() const { return m_bundleIdentifier; }
#endif
#endif

protected:
    explicit PaymentAuthorizationPresenter(Client& client)
        : m_client(client)
    {
    }

    virtual WKPaymentAuthorizationDelegate *platformDelegate() = 0;

#if PLATFORM(IOS_FAMILY) && ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
    String m_sceneIdentifier;
    String m_bundleIdentifier;
#endif

private:
    WeakPtr<Client> m_client;
};

} // namespace WebKit

#endif // USE(PASSKIT) && ENABLE(APPLE_PAY)
