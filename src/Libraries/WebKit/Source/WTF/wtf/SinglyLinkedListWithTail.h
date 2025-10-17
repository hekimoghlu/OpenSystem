/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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

#include <wtf/Assertions.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

template<typename T>
class SinglyLinkedListWithTail {
public:
    bool isEmpty() const { return !m_first; }
    
    template<typename SetNextFunc>
    void append(SetNextFunc&& setNextFunc, T* node)
    {
        if (!m_first) {
            RELEASE_ASSERT(!m_last);
            m_first = node;
            m_last = node;
            return;
        }
        
        std::forward<SetNextFunc>(setNextFunc)(m_last, node);
        m_last = node;
    }
    
    T* first() const { return m_first; }
    T* last() const { return m_last; }
    
private:
    T* m_first { nullptr };
    T* m_last { nullptr };
};

} // namespace WTF

using WTF::SinglyLinkedListWithTail;

