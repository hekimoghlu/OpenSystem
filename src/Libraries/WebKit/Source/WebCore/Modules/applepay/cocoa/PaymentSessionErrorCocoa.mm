/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#import "PaymentSessionError.h"

#if ENABLE(APPLE_PAY)

#import "ApplePaySessionError.h"
#import <pal/cocoa/PassKitSoftLink.h>

namespace WebCore {

static std::optional<ApplePaySessionError> additionalError(NSError *error)
{
#if HAVE(PASSKIT_INSTALLMENTS)
    // FIXME: Replace with PKPaymentErrorBindTokenUserInfoKey and
    // PKPaymentAuthorizationFeatureApplicationError when they're available in an SDK.
    static NSString * const bindTokenKey = @"PKPaymentErrorBindToken";
    static constexpr NSInteger pkPaymentAuthorizationFeatureApplicationError = -2016;

    if (error.code != pkPaymentAuthorizationFeatureApplicationError)
        return std::nullopt;

    id bindTokenValue = error.userInfo[bindTokenKey];
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(!bindTokenValue || [bindTokenValue isKindOfClass:NSString.class]);
    return ApplePaySessionError { "featureApplicationError"_s, { { "bindToken"_s, (NSString *)bindTokenValue } } };
#else
    UNUSED_PARAM(error);
    return std::nullopt;
#endif
}

PaymentSessionError::PaymentSessionError(RetainPtr<NSError>&& error)
    : m_platformError { WTFMove(error) }
{
}

ApplePaySessionError PaymentSessionError::sessionError() const
{
    ASSERT(!m_platformError || [[m_platformError domain] isEqualToString:PKPassKitErrorDomain]);

    if (auto error = additionalError(m_platformError.get()))
        return *error;

    return unknownError();
}

RetainPtr<NSError> PaymentSessionError::platformError() const
{
    return m_platformError;
}

ApplePaySessionError PaymentSessionError::unknownError() const
{
    return { "unknown"_s, { } };
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
