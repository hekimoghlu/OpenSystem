/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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
#import "ApplePaySetupFeatureWebCore.h"

#if ENABLE(APPLE_PAY)

#import "ApplePaySetupFeatureState.h"
#import "ApplePaySetupFeatureTypeWebCore.h"
#import <pal/spi/cocoa/PassKitSPI.h>

namespace WebCore {

bool ApplePaySetupFeature::supportsFeature(PKPaymentSetupFeature *feature)
{
    switch (feature.type) {
    case PKPaymentSetupFeatureTypeApplePay:
    case PKPaymentSetupFeatureTypeAppleCard:
        return true;

    default:
        return false;
    }
}

ApplePaySetupFeature::ApplePaySetupFeature() = default;
ApplePaySetupFeature::~ApplePaySetupFeature() = default;

ApplePaySetupFeatureType ApplePaySetupFeature::type() const
{
    switch ([m_feature type]) {
    case PKPaymentSetupFeatureTypeApplePay:
        return ApplePaySetupFeatureType::ApplePay;

    case PKPaymentSetupFeatureTypeAppleCard:
        return ApplePaySetupFeatureType::AppleCard;

    default:
        ASSERT(!supportsFeature(m_feature.get()));
        return ApplePaySetupFeatureType::ApplePay;
    }
}

ApplePaySetupFeatureState ApplePaySetupFeature::state() const
{
    switch ([m_feature state]) {
    case PKPaymentSetupFeatureStateUnsupported:
        return ApplePaySetupFeatureState::Unsupported;
    case PKPaymentSetupFeatureStateSupported:
        return ApplePaySetupFeatureState::Supported;
    case PKPaymentSetupFeatureStateSupplementarySupported:
        return ApplePaySetupFeatureState::SupplementarySupported;
    case PKPaymentSetupFeatureStateCompleted:
        return ApplePaySetupFeatureState::Completed;
    }
}

#if ENABLE(APPLE_PAY_INSTALLMENTS)
bool ApplePaySetupFeature::supportsInstallments() const
{
#if PLATFORM(MAC)
    if (![m_feature respondsToSelector:@selector(supportedOptions)])
        return false;
#endif
    return [m_feature supportedOptions] & PKPaymentSetupFeatureSupportedOptionsInstallments;
}
#endif

ApplePaySetupFeature::ApplePaySetupFeature(PKPaymentSetupFeature *feature)
    : m_feature { feature }
{
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
