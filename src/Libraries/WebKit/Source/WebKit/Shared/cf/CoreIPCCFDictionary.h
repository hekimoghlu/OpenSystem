/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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

class CoreIPCCFDictionary {
public:
    using KeyValueVector = Vector<KeyValuePair<CoreIPCCFType, CoreIPCCFType>>;

    CoreIPCCFDictionary(CFDictionaryRef);
    CoreIPCCFDictionary(CoreIPCCFDictionary&&);
    CoreIPCCFDictionary(std::unique_ptr<KeyValueVector>&&);
    ~CoreIPCCFDictionary();
    RetainPtr<CFDictionaryRef> createCFDictionary() const;
    const std::unique_ptr<KeyValueVector>& vector() const { return m_vector; }
private:
    std::unique_ptr<KeyValueVector> m_vector;
};

} // namespace WebKit

#endif // USE(CF)
