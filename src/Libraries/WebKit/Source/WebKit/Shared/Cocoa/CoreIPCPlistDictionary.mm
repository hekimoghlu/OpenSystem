/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#import "CoreIPCPlistDictionary.h"

#if PLATFORM(COCOA)

#import "CoreIPCData.h"
#import "CoreIPCDate.h"
#import "CoreIPCNumber.h"
#import "CoreIPCPlistArray.h"
#import "CoreIPCPlistObject.h"
#import "CoreIPCString.h"
#import <wtf/Assertions.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CoreIPCPlistDictionary);

CoreIPCPlistDictionary::CoreIPCPlistDictionary(NSDictionary *dictionary)
{
    m_keyValuePairs.reserveInitialCapacity(dictionary.count);

    for (id key in dictionary) {
        id value = dictionary[key];

        if (!key || ![key isKindOfClass:[NSString class]]) {
            ASSERT_NOT_REACHED();
            continue;
        }

        if (!CoreIPCPlistObject::isPlistType(value)) {
            ASSERT_NOT_REACHED();
            continue;
        }

        m_keyValuePairs.append({ CoreIPCString(key), CoreIPCPlistObject(value) });
    }
}

CoreIPCPlistDictionary::CoreIPCPlistDictionary(const RetainPtr<NSDictionary>& dictionary)
    : CoreIPCPlistDictionary(dictionary.get()) { }

CoreIPCPlistDictionary::CoreIPCPlistDictionary(CoreIPCPlistDictionary&&) = default;

CoreIPCPlistDictionary::~CoreIPCPlistDictionary() = default;

CoreIPCPlistDictionary::CoreIPCPlistDictionary(ValueType&& keyValuePairs)
    : m_keyValuePairs(WTFMove(keyValuePairs)) { }

RetainPtr<id> CoreIPCPlistDictionary::toID() const
{
    auto result = adoptNS([[NSMutableDictionary alloc] initWithCapacity:m_keyValuePairs.size()]);
    for (auto& keyValuePair : m_keyValuePairs) {
        auto key = keyValuePair.key.toID();
        auto value = keyValuePair.value.toID();
        if (key && value)
            [result setObject:value.get() forKey:key.get()];
    }
    return result;
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
