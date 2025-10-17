/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
#import "PaymentMerchantSession.h"

#if ENABLE(APPLE_PAY)

#import <JavaScriptCore/JSONObject.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebCore {

std::optional<PaymentMerchantSession> PaymentMerchantSession::fromJS(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, String&)
{
    // FIXME: Don't round-trip using NSString.
    auto jsonString = JSONStringify(&lexicalGlobalObject, value, 0);
    if (!jsonString)
        return std::nullopt;

    auto dictionary = dynamic_objc_cast<NSDictionary>([NSJSONSerialization JSONObjectWithData:[(NSString *)jsonString dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil]);
    if (!dictionary || ![dictionary isKindOfClass:[NSDictionary class]])
        return std::nullopt;

    auto pkPaymentMerchantSession = adoptNS([PAL::allocPKPaymentMerchantSessionInstance() initWithDictionary:dictionary]);

    return PaymentMerchantSession(pkPaymentMerchantSession.get());
}

}

#endif
