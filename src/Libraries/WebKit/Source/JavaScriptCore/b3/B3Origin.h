/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

#if ENABLE(B3_JIT)

#include <wtf/PrintStream.h>

namespace JSC { namespace B3 {

// Whoever generates B3IR can choose to put origins on values. When you do this, B3 will be able to
// account, down to the machine code, which instruction corresponds to which origin. B3
// transformations must preserve Origins carefully. It's an error to write a transformation that
// either drops Origins or lies about them.
class Origin {
public:
    explicit Origin(const void* data = nullptr)
        : m_data(data)
    {
    }

    explicit operator bool() const { return !!m_data; }

    const void* data() const { return m_data; }

    friend bool operator==(const Origin&, const Origin&) = default;

    // You should avoid using this. Use OriginDump instead.
    void dump(PrintStream&) const;
    
private:
    const void* m_data;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
