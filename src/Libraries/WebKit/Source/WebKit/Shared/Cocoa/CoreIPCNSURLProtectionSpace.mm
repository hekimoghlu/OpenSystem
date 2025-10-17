/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#import "CoreIPCNSURLProtectionSpace.h"

#import "ArgumentCoders.h"
#import <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA) && HAVE(WK_SECURE_CODING_NSURLPROTECTIONSPACE)

@interface NSURLProtectionSpace (WKSecureCoding)
- (NSDictionary *)_webKitPropertyListData;
- (instancetype)_initWithWebKitPropertyListData:(NSDictionary *)plist;
@end

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CoreIPCNSURLProtectionSpace);

#define SET_OBJECT(NAME, CLASS, WRAPPER)    \
    id NAME = dict[@#NAME];                 \
    if ([NAME isKindOfClass:CLASS.class]) { \
        auto var = WRAPPER(NAME);           \
        m_data.NAME = WTFMove(var);         \
    }

CoreIPCNSURLProtectionSpace::CoreIPCNSURLProtectionSpace(NSURLProtectionSpace *ps)
{
    auto dict = [ps _webKitPropertyListData];

    SET_OBJECT(host, NSString, CoreIPCString);

    id port = dict[@"port"];
    if ([port isKindOfClass:NSNumber.class])
        m_data.port = [port unsignedShortValue];

    id type = dict[@"type"];
    if ([type isKindOfClass:NSNumber.class]) {
        auto val = [type unsignedCharValue];
        if (isValidEnum<WebCore::ProtectionSpaceBaseServerType>(val))
            m_data.type = static_cast<WebCore::ProtectionSpaceBaseServerType>(val);
    }

    SET_OBJECT(realm, NSString, CoreIPCString);

    id scheme = dict[@"scheme"];
    if ([scheme isKindOfClass:NSNumber.class]) {
        auto val = [scheme unsignedCharValue];
        if (isValidEnum<WebCore::ProtectionSpaceBaseAuthenticationScheme>(val))
            m_data.scheme = static_cast<WebCore::ProtectionSpaceBaseAuthenticationScheme>(val);
    }

    id trust = dict[@"trust"];
    if (trust && CFGetTypeID((CFTypeRef)trust) == SecTrustGetTypeID())
        m_data.trust = { CoreIPCSecTrust((SecTrustRef)trust) };

    NSArray *distnames = dict[@"distnames"];
    if ([distnames isKindOfClass:NSArray.class]) {
        bool allElementsValid = true;
        Vector<WebKit::CoreIPCData> data;
        data.reserveInitialCapacity(distnames.count);
        for (NSData *d in distnames) {
            if (![d isKindOfClass:NSData.class]) {
                allElementsValid = false;
                break;
            }
            data.append(d);
        }
        if (allElementsValid)
            m_data.distnames = WTFMove(data);
    }
}

CoreIPCNSURLProtectionSpace::CoreIPCNSURLProtectionSpace(CoreIPCNSURLProtectionSpaceData&& data)
    : m_data(WTFMove(data)) { }

CoreIPCNSURLProtectionSpace::CoreIPCNSURLProtectionSpace(const RetainPtr<NSURLProtectionSpace>& ps)
    : CoreIPCNSURLProtectionSpace(ps.get()) { }

RetainPtr<id> CoreIPCNSURLProtectionSpace::toID() const
{
    auto dict = adoptNS([[NSMutableDictionary alloc] initWithCapacity:7]); // Initialized with the count of members in CoreIPCNSURLProtectionSpaceData

    if (m_data.host)
        [dict setObject:m_data.host->toID().get() forKey:@"host"];

    [dict setObject:[NSNumber numberWithUnsignedShort:m_data.port] forKey:@"port"];
    [dict setObject:[NSNumber numberWithUnsignedChar:static_cast<uint8_t>(m_data.type)] forKey:@"type"];

    if (m_data.realm)
        [dict setObject:m_data.realm->toID().get() forKey:@"realm"];

    [dict setObject:[NSNumber numberWithUnsignedChar:static_cast<uint8_t>(m_data.scheme)] forKey:@"scheme"];

    if (m_data.trust)
        [dict setObject:(id)m_data.trust->createSecTrust().get() forKey:@"trust"];

    if (m_data.distnames) {
        auto array = adoptNS([[NSMutableArray alloc] initWithCapacity:m_data.distnames->size()]);
        for (auto& value : *m_data.distnames)
            [array addObject:value.toID().get()];
        [dict setObject:array.get() forKey:@"distnames"];
    }
    return adoptNS([[NSURLProtectionSpace alloc] _initWithWebKitPropertyListData:dict.get()]);
}

#undef SET_OBJECT

} // namespace WebKit

#endif
