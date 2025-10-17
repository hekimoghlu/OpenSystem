/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
#import "CoreIPCPlistArray.h"

#if PLATFORM(COCOA)

#import "CoreIPCData.h"
#import "CoreIPCDate.h"
#import "CoreIPCNumber.h"
#import "CoreIPCPlistDictionary.h"
#import "CoreIPCPlistObject.h"
#import "CoreIPCString.h"
#import <wtf/cocoa/VectorCocoa.h>

namespace WebKit {

CoreIPCPlistArray::CoreIPCPlistArray(NSArray *array)
{
    for (id value in array) {
        if (!CoreIPCPlistObject::isPlistType(value))
            continue;
        m_array.append(CoreIPCPlistObject(value));
    }
}

CoreIPCPlistArray::CoreIPCPlistArray(const RetainPtr<NSArray>& array)
    : CoreIPCPlistArray(array.get()) { }

CoreIPCPlistArray::CoreIPCPlistArray(CoreIPCPlistArray&&) = default;

CoreIPCPlistArray::~CoreIPCPlistArray() = default;

CoreIPCPlistArray::CoreIPCPlistArray(Vector<CoreIPCPlistObject>&& array)
    : m_array(WTFMove(array)) { }

RetainPtr<id> CoreIPCPlistArray::toID() const
{
    return createNSArray(m_array, [] (auto& object) {
        return object.toID();
    });
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
