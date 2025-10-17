/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#import "ApplePayPaymentSetupFeaturesWebKit.h"

#if ENABLE(APPLE_PAY)

#import "ArgumentCodersCocoa.h"
#import "Decoder.h"
#import "Encoder.h"
#import <WebCore/ApplePaySetupFeatureWebCore.h>

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebKit {

static NSArray<PKPaymentSetupFeature *> *toPlatformFeatures(Vector<Ref<WebCore::ApplePaySetupFeature>>&& features)
{
    NSMutableArray *platformFeatures = [NSMutableArray arrayWithCapacity:features.size()];
    for (auto& feature : features) {
        [platformFeatures addObject:feature->platformFeature()];
    }
    return platformFeatures;
}

PaymentSetupFeatures::PaymentSetupFeatures(Vector<Ref<WebCore::ApplePaySetupFeature>>&& features)
    : m_platformFeatures { toPlatformFeatures(WTFMove(features)) }
{
}

PaymentSetupFeatures::PaymentSetupFeatures(RetainPtr<NSArray>&& platformFeatures)
    : m_platformFeatures { WTFMove(platformFeatures) }
{
}

PaymentSetupFeatures::operator Vector<Ref<WebCore::ApplePaySetupFeature>>() const
{
    Vector<Ref<WebCore::ApplePaySetupFeature>> features;
    features.reserveInitialCapacity([m_platformFeatures count]);
    for (PKPaymentSetupFeature *platformFeature in m_platformFeatures.get()) {
        if (WebCore::ApplePaySetupFeature::supportsFeature(platformFeature))
            features.append(WebCore::ApplePaySetupFeature::create(platformFeature));
    }
    return features;
}

} // namespace WebKit

#endif // ENABLE(APPLE_PAY)
