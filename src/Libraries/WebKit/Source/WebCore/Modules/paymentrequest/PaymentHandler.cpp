/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#include "PaymentHandler.h"

#if ENABLE(PAYMENT_REQUEST)

#if ENABLE(APPLE_PAY_AMS_UI)
#include "ApplePayAMSUIPaymentHandler.h"
#endif

#if ENABLE(APPLE_PAY)
#include "ApplePayPaymentHandler.h"
#endif

namespace WebCore {

RefPtr<PaymentHandler> PaymentHandler::create(Document& document, PaymentRequest& paymentRequest, const PaymentRequest::MethodIdentifier& identifier)
{
#if ENABLE(APPLE_PAY)
    if (ApplePayPaymentHandler::handlesIdentifier(identifier))
        return adoptRef(new ApplePayPaymentHandler(document, identifier, paymentRequest));
#endif

#if ENABLE(APPLE_PAY_AMS_UI)
    if (ApplePayAMSUIPaymentHandler::handlesIdentifier(identifier))
        return adoptRef(new ApplePayAMSUIPaymentHandler(document, identifier, paymentRequest));
#endif

    UNUSED_PARAM(document);
    UNUSED_PARAM(paymentRequest);
    UNUSED_PARAM(identifier);
    return nullptr;
}

ExceptionOr<void> PaymentHandler::canCreateSession(Document& document)
{
#if ENABLE(APPLE_PAY)
    auto result = PaymentSession::canCreateSession(document);
    if (result.hasException())
        return Exception { ExceptionCode::SecurityError, result.releaseException().releaseMessage() };
#else
    UNUSED_PARAM(document);
#endif

    return { };
}

ExceptionOr<void> PaymentHandler::validateData(Document& document, JSC::JSValue data, const PaymentRequest::MethodIdentifier& identifier)
{
#if ENABLE(APPLE_PAY)
    if (ApplePayPaymentHandler::handlesIdentifier(identifier))
        return ApplePayPaymentHandler::validateData(document, data);
#endif

#if ENABLE(APPLE_PAY_AMS_UI)
    if (ApplePayAMSUIPaymentHandler::handlesIdentifier(identifier))
        return ApplePayAMSUIPaymentHandler::validateData(document, data);
#endif

    UNUSED_PARAM(document);
    UNUSED_PARAM(data);
    UNUSED_PARAM(identifier);
    return { };
}

bool PaymentHandler::hasActiveSession(Document& document)
{
#if ENABLE(APPLE_PAY)
    if (ApplePayPaymentHandler::hasActiveSession(document))
        return true;
#endif

#if ENABLE(APPLE_PAY_AMS_UI)
    if (ApplePayAMSUIPaymentHandler::hasActiveSession(document))
        return true;
#endif

    UNUSED_PARAM(document);
    return false;
}

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
