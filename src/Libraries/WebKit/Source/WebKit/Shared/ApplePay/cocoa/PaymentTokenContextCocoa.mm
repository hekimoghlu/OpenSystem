/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#import "config.h"
#import "PaymentTokenContext.h"

#if HAVE(PASSKIT_MULTI_MERCHANT_PAYMENTS)

#import "WebPaymentCoordinatorProxyCocoa.h"
#import <WebCore/ApplePayPaymentTokenContext.h>
#import <WebCore/PaymentHeaders.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/text/WTFString.h>

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebKit {
using namespace WebCore;

RetainPtr<PKPaymentTokenContext> platformPaymentTokenContext(const ApplePayPaymentTokenContext& webTokenContext)
{
    RetainPtr<NSString> merchantDomain;
    if (!webTokenContext.merchantDomain.isNull())
        merchantDomain = webTokenContext.merchantDomain;
    return adoptNS([PAL::allocPKPaymentTokenContextInstance() initWithMerchantIdentifier:webTokenContext.merchantIdentifier externalIdentifier:webTokenContext.externalIdentifier merchantName:webTokenContext.merchantName merchantDomain:merchantDomain.get() amount:WebCore::toDecimalNumber(webTokenContext.amount)]);
}

RetainPtr<NSArray<PKPaymentTokenContext *>> platformPaymentTokenContexts(const Vector<ApplePayPaymentTokenContext>& webTokenContexts)
{
    return createNSArray(webTokenContexts, platformPaymentTokenContext);
}

} // namespace WebKit

#endif // HAVE(PASSKIT_MULTI_MERCHANT_PAYMENTS)
