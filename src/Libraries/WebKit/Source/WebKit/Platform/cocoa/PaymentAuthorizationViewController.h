/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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

#include "PaymentAuthorizationPresenter.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS PKPaymentAuthorizationViewController;
OBJC_CLASS PKPaymentRequest;
OBJC_CLASS WKPaymentAuthorizationDelegate;
OBJC_CLASS WKPaymentAuthorizationViewControllerDelegate;

namespace WebKit {

class WebPaymentCoordinatorProxy;

class PaymentAuthorizationViewController final : public PaymentAuthorizationPresenter {
public:
    static Ref<PaymentAuthorizationViewController> create(PaymentAuthorizationPresenter::Client&, PKPaymentRequest *, PKPaymentAuthorizationViewController * = nil);

private:
    PaymentAuthorizationViewController(PaymentAuthorizationPresenter::Client&, PKPaymentRequest *, PKPaymentAuthorizationViewController * = nil);

    // PaymentAuthorizationPresenter
    WKPaymentAuthorizationDelegate *platformDelegate() final;
    void dismiss() final;
#if PLATFORM(IOS_FAMILY)
    void present(UIViewController *, CompletionHandler<void(bool)>&&) final;
#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
    void presentInScene(const String& sceneIdentifier, const String& bundleIdentifier, CompletionHandler<void(bool)>&&) final;
#endif
#endif

    RetainPtr<PKPaymentAuthorizationViewController> m_viewController;
    RetainPtr<WKPaymentAuthorizationViewControllerDelegate> m_delegate;
};

} // namespace WebKit

#endif // USE(PASSKIT) && ENABLE(APPLE_PAY)
