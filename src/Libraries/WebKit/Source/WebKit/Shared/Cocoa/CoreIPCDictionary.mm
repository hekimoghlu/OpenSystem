/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#import "CoreIPCDictionary.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)

#import "CoreIPCTypes.h"

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CoreIPCDictionary);

CoreIPCDictionary::CoreIPCDictionary(NSDictionary *dictionary)
{
    m_keyValuePairs.reserveInitialCapacity(dictionary.count);

    for (id key in dictionary) {
        id value = dictionary[key];
        ASSERT(value);

        // Ignore values we don't support.
        ASSERT(IPC::isSerializableValue(key));
        ASSERT(IPC::isSerializableValue(value));
        if (!IPC::isSerializableValue(key) || !IPC::isSerializableValue(value))
            continue;

        m_keyValuePairs.append({ CoreIPCNSCFObject(key), CoreIPCNSCFObject(value) });
    }
}

CoreIPCDictionary::CoreIPCDictionary(const RetainPtr<NSDictionary>& dictionary)
    : CoreIPCDictionary(dictionary.get()) { }

CoreIPCDictionary::CoreIPCDictionary(CoreIPCDictionary&&) = default;

CoreIPCDictionary::~CoreIPCDictionary() = default;

CoreIPCDictionary::CoreIPCDictionary(ValueType&& keyValuePairs)
    : m_keyValuePairs(WTFMove(keyValuePairs)) { }

RetainPtr<id> CoreIPCDictionary::toID() const
{
    auto result = adoptNS([[NSMutableDictionary alloc] initWithCapacity:m_keyValuePairs.size()]);
    for (auto& keyValuePair : m_keyValuePairs)
        [result setObject:keyValuePair.value.toID().get() forKey:keyValuePair.key.toID().get()];
    return result;
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
