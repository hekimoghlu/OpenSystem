/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#import "CoreIPCArray.h"

#if PLATFORM(COCOA)

#import "CoreIPCNSCFObject.h"
#import "CoreIPCTypes.h"

namespace WebKit {

CoreIPCArray::CoreIPCArray(NSArray *array)
{
    for (id value in array) {
        if (!IPC::isSerializableValue(value))
            continue;
        m_array.append(CoreIPCNSCFObject(value));
    }
}

CoreIPCArray::CoreIPCArray(const RetainPtr<NSArray>& array)
    : CoreIPCArray(array.get()) { }

CoreIPCArray::CoreIPCArray(CoreIPCArray&&) = default;

CoreIPCArray::~CoreIPCArray() = default;

CoreIPCArray::CoreIPCArray(Vector<CoreIPCNSCFObject>&& array)
    : m_array(WTFMove(array)) { }

RetainPtr<id> CoreIPCArray::toID() const
{
    auto result = adoptNS([[NSMutableArray alloc] initWithCapacity:m_array.size()]);
    for (auto& object : m_array)
        [result addObject:object.toID().get()];
    return result;
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
