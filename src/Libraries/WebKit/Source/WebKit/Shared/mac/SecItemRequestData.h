/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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

#include "ArgumentCoders.h"
#include <wtf/RetainPtr.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {
    
class SecItemRequestData {
public:
    enum class Type : uint8_t {
        Invalid,
        CopyMatching,
        Add,
        Update,
        Delete,
    };

    SecItemRequestData() = default;
    SecItemRequestData(Type, CFDictionaryRef query);
    SecItemRequestData(Type, CFDictionaryRef query, CFDictionaryRef attributesToMatch);
    SecItemRequestData(Type, RetainPtr<CFDictionaryRef>&& query, RetainPtr<CFDictionaryRef>&& attributesToMatch);

    Type type() const { return m_type; }

    CFDictionaryRef query() const { return m_queryDictionary.get(); }
    CFDictionaryRef attributesToMatch() const { return m_attributesToMatch.get(); }

private:
    friend struct IPC::ArgumentCoder<WebKit::SecItemRequestData, void>;

    Type m_type { Type::Invalid };
    RetainPtr<CFDictionaryRef> m_queryDictionary;
    RetainPtr<CFDictionaryRef> m_attributesToMatch;
};
    
} // namespace WebKit
