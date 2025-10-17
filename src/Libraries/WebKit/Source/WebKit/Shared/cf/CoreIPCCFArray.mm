/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#import "CoreIPCCFArray.h"

#if USE(CF)

#import "CoreIPCCFType.h"
#import "CoreIPCTypes.h"
#import <wtf/cf/VectorCF.h>

namespace WebKit {

CoreIPCCFArray::CoreIPCCFArray(Vector<CoreIPCCFType>&& array)
    : m_array(WTFMove(array)) { }

CoreIPCCFArray::~CoreIPCCFArray() = default;

CoreIPCCFArray::CoreIPCCFArray(CoreIPCCFArray&&) = default;

CoreIPCCFArray::CoreIPCCFArray(CFArrayRef array)
{
    CFIndex count = array ? CFArrayGetCount(array) : 0;
    for (CFIndex i = 0; i < count; i++) {
        CFTypeRef element = CFArrayGetValueAtIndex(array, i);
        if (IPC::typeFromCFTypeRef(element) == IPC::CFType::Unknown)
            continue;
        m_array.append(CoreIPCCFType(element));
    }
}

RetainPtr<CFArrayRef> CoreIPCCFArray::createCFArray() const
{
    return WTF::createCFArray(m_array, [] (const CoreIPCCFType& element) {
        return element.toID();
    });
}

} // namespace WebKit

#endif // USE(CF)
