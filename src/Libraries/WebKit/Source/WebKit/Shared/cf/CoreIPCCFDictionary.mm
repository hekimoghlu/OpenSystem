/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#import "CoreIPCCFDictionary.h"

#if USE(CF)

#import "CoreIPCCFType.h"
#import "CoreIPCTypes.h"

namespace WebKit {

CoreIPCCFDictionary::CoreIPCCFDictionary(std::unique_ptr<KeyValueVector>&& vector)
    : m_vector(WTFMove(vector)) { }

CoreIPCCFDictionary::CoreIPCCFDictionary(CoreIPCCFDictionary&&) = default;

CoreIPCCFDictionary::~CoreIPCCFDictionary() = default;

CoreIPCCFDictionary::CoreIPCCFDictionary(CFDictionaryRef dictionary)
{
    if (!dictionary)
        return;
    m_vector = makeUnique<KeyValueVector>();
    m_vector->reserveInitialCapacity(CFDictionaryGetCount(dictionary));
    [(__bridge NSDictionary *)dictionary enumerateKeysAndObjectsUsingBlock:^(id key, id value, BOOL*) {
        if (IPC::typeFromCFTypeRef(key) == IPC::CFType::Unknown)
            return;
        if (IPC::typeFromCFTypeRef(value) == IPC::CFType::Unknown)
            return;
        m_vector->append({ CoreIPCCFType(key), CoreIPCCFType(value) });
    }];
}

RetainPtr<CFDictionaryRef> CoreIPCCFDictionary::createCFDictionary() const
{
    if (!m_vector)
        return nil;

    auto result = adoptNS([[NSMutableDictionary alloc] initWithCapacity:m_vector->size()]);
    for (auto& pair : *m_vector) {
        auto key = pair.key.toID();
        auto value = pair.value.toID();
        if (key && value)
            [result setObject:value.get() forKey:key.get()];
    }
    return (__bridge CFDictionaryRef)result.get();
}

} // namespace WebKit

#endif // USE(CF)
