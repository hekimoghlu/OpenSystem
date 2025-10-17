/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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

#if USE(PASSKIT) && PLATFORM(IOS_FAMILY)

#include "PaymentAuthorizationPresenter.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS PKPaymentAuthorizationController;
OBJC_CLASS PKPaymentRequest;
OBJC_CLASS WKPaymentAuthorizationControllerDelegate;
OBJC_CLASS WKPaymentAuthorizationDelegate;

namespace WebKit {

class PaymentAuthorizationController final : public PaymentAuthorizationPresenter {
public:
    static Ref<PaymentAuthorizationController> create(PaymentAuthorizationPresenter::Client&, PKPaymentRequest *);

private:
    PaymentAuthorizationController(PaymentAuthorizationPresenter::Client&, PKPaymentRequest *);

    // PaymentAuthorizationPresenter
    WKPaymentAuthorizationDelegate *platformDelegate() final;
    void dismiss() final;
    void present(UIViewController *, CompletionHandler<void(bool)>&&) final;
#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
    void presentInScene(const String& sceneIdentifier, const String& bundleIdentifier, CompletionHandler<void(bool)>&&) final;
#endif

    RetainPtr<PKPaymentAuthorizationController> m_controller;
    RetainPtr<WKPaymentAuthorizationControllerDelegate> m_delegate;
};

} // namespace WebKit

#endif // USE(PASSKIT) && PLATFORM(IOS_FAMILY)
