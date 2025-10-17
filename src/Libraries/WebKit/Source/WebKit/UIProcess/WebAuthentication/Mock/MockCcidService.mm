/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#include "MockCcidService.h"

#if ENABLE(WEB_AUTHN)

#import <CryptoTokenKit/TKSmartCard.h>
#include <wtf/RunLoop.h>

@interface _WKMockTKSmartCard : TKSmartCard
- (instancetype)initWithService:(WeakPtr<WebKit::MockCcidService>&&)service;
@end

@implementation _WKMockTKSmartCard {
    WeakPtr<WebKit::MockCcidService> m_service;
}

- (instancetype)initWithService:(WeakPtr<WebKit::MockCcidService>&&)service
{
    if (!(self = [super init]))
        return nil;

    m_service = WTFMove(service);

    return self;
}

- (void)beginSessionWithReply:(void(^)(BOOL success, NSError * error))reply
{
    reply(TRUE, nil);
}

- (void)endSession
{
}

- (void)transmitRequest:(NSData *)request reply:(void(^)(NSData * response, NSError * error))reply
{
    reply(Ref { *m_service }->nextReply().get(), nil);
}

@end

namespace WebKit {

Ref<MockCcidService> MockCcidService::create(AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
{
    return adoptRef(*new MockCcidService(observer, configuration));
}

MockCcidService::MockCcidService(AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
    : CcidService(observer)
    , m_configuration(configuration)
{
}

void MockCcidService::platformStartDiscovery()
{
    if (!!m_configuration.ccid) {
        auto card = adoptNS([[_WKMockTKSmartCard alloc] initWithService:this]);
        onValidCard(WTFMove(card));
        return;
    }
    LOG_ERROR("No ccid authenticators is available.");
}

RetainPtr<NSData> MockCcidService::nextReply()
{
    if (m_configuration.ccid->payloadBase64.isEmpty())
        return nil;

    auto result = adoptNS([[NSData alloc] initWithBase64EncodedString:m_configuration.ccid->payloadBase64[0] options:NSDataBase64DecodingIgnoreUnknownCharacters]);
    m_configuration.ccid->payloadBase64.remove(0);
    return result;
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
