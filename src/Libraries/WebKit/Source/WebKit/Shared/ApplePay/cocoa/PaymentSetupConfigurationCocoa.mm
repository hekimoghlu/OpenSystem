/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
#import "PaymentSetupConfigurationWebKit.h"

#if ENABLE(APPLE_PAY)

#import "ArgumentCodersCocoa.h"
#import "Decoder.h"
#import "Encoder.h"
#import <WebCore/ApplePaySetupConfiguration.h>
#import <wtf/URL.h>

#import <pal/cocoa/PassKitSoftLink.h>

@interface PKPaymentSetupConfiguration (WebKit)
@property (nonatomic, copy) NSArray<NSString *> *signedFields;
@end

namespace WebKit {

static RetainPtr<PKPaymentSetupConfiguration> toPlatformConfiguration(const WebCore::ApplePaySetupConfiguration& coreConfiguration, const URL& url)
{
#if PLATFORM(MAC)
    if (!PAL::getPKPaymentSetupConfigurationClass())
        return nil;
#endif

    auto configuration = adoptNS([PAL::allocPKPaymentSetupConfigurationInstance() init]);
    [configuration setMerchantIdentifier:coreConfiguration.merchantIdentifier];
    [configuration setOriginatingURL:url];
    [configuration setReferrerIdentifier:coreConfiguration.referrerIdentifier];
ALLOW_NEW_API_WITHOUT_GUARDS_BEGIN
    [configuration setSignature:coreConfiguration.signature];
ALLOW_NEW_API_WITHOUT_GUARDS_END

    if ([configuration respondsToSelector:@selector(setSignedFields:)]) {
        NSMutableArray *signedFields = [NSMutableArray arrayWithCapacity:coreConfiguration.signedFields.size()];
        for (auto& signedField : coreConfiguration.signedFields)
            [signedFields addObject:signedField];
        [configuration setSignedFields:signedFields];
    }

    return configuration;
}

RetainPtr<PKPaymentSetupConfiguration> PaymentSetupConfiguration::platformConfiguration() const
{
    return toPlatformConfiguration(m_configuration, m_url);
}

PaymentSetupConfiguration::PaymentSetupConfiguration(const WebCore::ApplePaySetupConfiguration& configuration, const URL& url)
    : m_configuration(configuration)
    , m_url(url)
{
}

} // namespace WebKit

#endif // ENABLE(APPLE_PAY)
