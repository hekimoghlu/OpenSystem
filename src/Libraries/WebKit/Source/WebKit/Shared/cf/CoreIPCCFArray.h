/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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
#pragma once

#if USE(CF)

#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>

namespace WebKit {

class CoreIPCCFType;

class CoreIPCCFArray {
public:
    CoreIPCCFArray(CFArrayRef);
    CoreIPCCFArray(Vector<CoreIPCCFType>&&);
    CoreIPCCFArray(CoreIPCCFArray&&);
    ~CoreIPCCFArray();
    RetainPtr<CFArrayRef> createCFArray() const;
    const Vector<CoreIPCCFType>& array() const { return m_array; }
private:
    Vector<CoreIPCCFType> m_array;
};

} // namespace WebKit

#endif // USE(CF)
