/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
#import "DisbursementRequest.h"

#if HAVE(PASSKIT_DISBURSEMENTS)

#import "WebPaymentCoordinatorProxyCocoa.h"

#import <WebCore/ApplePayContactField.h>
#import <WebCore/ApplePayDisbursementRequest.h>
#import <WebCore/ApplePaySessionPaymentRequest.h>
#import <WebCore/PaymentSummaryItems.h>

#import <pal/spi/cocoa/PassKitSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/VectorCocoa.h>

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebKit {
using namespace WebCore;

RetainPtr<PKDisbursementPaymentRequest> platformDisbursementRequest(const ApplePaySessionPaymentRequest& request, const URL& originatingURL, const std::optional<Vector<ApplePayContactField>>& requiredRecipientContactFields)
{
    // This merchantID is not actually used for web payments, passing an empty string here is fine
    auto disbursementRequest = adoptNS([PAL::allocPKDisbursementRequestInstance() initWithMerchantIdentifier:@"" currencyCode:request.currencyCode() regionCode:request.countryCode() supportedNetworks:createNSArray(request.supportedNetworks()).get() merchantCapabilities:toPKMerchantCapabilities(request.merchantCapabilities()) summaryItems:WebCore::platformDisbursementSummaryItems(request.lineItems())]);

    // FIXME: we should consolidate the types for various contact fields in the system(WebCore::ApplePayContactField, WebCore::ApplePaySessionPaymentRequest::ContactFields etc.)
    if (requiredRecipientContactFields) {
        NSMutableArray<NSString *> *result = [NSMutableArray array];
        for (auto& contactField : requiredRecipientContactFields.value()) {
            switch (contactField) {
            case ApplePayContactField::Email:
                [result addObject:PKContactFieldEmailAddress];
                break;
            case ApplePayContactField::Name:
                [result addObject:PKContactFieldName];
                break;
            case ApplePayContactField::PhoneticName:
                [result addObject:PKContactFieldPhoneticName];
                break;
            case ApplePayContactField::Phone:
                [result addObject:PKContactFieldPhoneNumber];
                break;
            case ApplePayContactField::PostalAddress:
                [result addObject:PKContactFieldPostalAddress];
                break;
            }
        }

        [disbursementRequest setRequiredRecipientContactFields:[result copy]];
    }

    auto disbursementPaymentRequest = adoptNS([PAL::allocPKDisbursementPaymentRequestInstance() initWithDisbursementRequest:disbursementRequest.get()]);
    [disbursementPaymentRequest setOriginatingURL:originatingURL];
    [disbursementPaymentRequest setAPIType:PKPaymentRequestAPITypeWebPaymentRequest];
    return disbursementPaymentRequest;
}

} // namespace WebKit

#endif // HAVE(PASSKIT_DISBURSEMENTS)
