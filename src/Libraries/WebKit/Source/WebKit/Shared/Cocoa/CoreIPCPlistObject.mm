/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
#import "CoreIPCPlistObject.h"

#if PLATFORM(COCOA)

#import "ArgumentCodersCocoa.h"
#import "CoreIPCData.h"
#import "CoreIPCDate.h"
#import "CoreIPCNumber.h"
#import "CoreIPCPlistArray.h"
#import "CoreIPCPlistDictionary.h"
#import "CoreIPCString.h"
#import "GeneratedWebKitSecureCoding.h"
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebKit {

bool CoreIPCPlistObject::isPlistType(id value)
{
    if ([value isKindOfClass:NSString.class]
        || [value isKindOfClass:NSArray.class]
        || [value isKindOfClass:NSData.class]
        || [value isKindOfClass:NSNumber.class]
        || [value isKindOfClass:NSDate.class]
        || [value isKindOfClass:NSDictionary.class])
        return true;
    return false;
}

static PlistValue valueFromID(id object)
{
    switch (IPC::typeFromObject(object)) {
    case IPC::NSType::Array:
        return CoreIPCPlistArray((NSArray *)object);
    case IPC::NSType::Data:
        return CoreIPCData((NSData *)object);
    case IPC::NSType::Date:
        return CoreIPCDate(bridge_cast((NSDate *)object));
    case IPC::NSType::Dictionary:
        return CoreIPCPlistDictionary((NSDictionary *)object);
    case IPC::NSType::Number:
        return CoreIPCNumber(bridge_cast((NSNumber *)object));
    case IPC::NSType::String:
        return CoreIPCString((NSString *)object);
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

CoreIPCPlistObject::CoreIPCPlistObject(id object)
    : m_value(makeUniqueRefWithoutFastMallocCheck<PlistValue>(valueFromID(object)))
{
}

CoreIPCPlistObject::CoreIPCPlistObject(UniqueRef<PlistValue>&& value)
    : m_value(WTFMove(value))
{
}

RetainPtr<id> CoreIPCPlistObject::toID() const
{
    return WTF::switchOn(*m_value, [&](auto& object) {
        return object.toID();
    });
}

} // namespace WebKit

namespace IPC {

void ArgumentCoder<UniqueRef<WebKit::PlistValue>>::encode(Encoder& encoder, const UniqueRef<WebKit::PlistValue>& object)
{
    encoder << *object;
}

std::optional<UniqueRef<WebKit::PlistValue>> ArgumentCoder<UniqueRef<WebKit::PlistValue>>::decode(Decoder& decoder)
{
    auto object = decoder.decode<WebKit::PlistValue>();
    if (!object)
        return std::nullopt;
    return makeUniqueRefWithoutFastMallocCheck<WebKit::PlistValue>(WTFMove(*object));
}

} // namespace IPC

#endif // PLATFORM(COCOA)
