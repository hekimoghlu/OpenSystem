/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

#include <wtf/Vector.h>

namespace IPC {

template<typename... Types>
class ArrayReferenceTuple {
public:
    ArrayReferenceTuple() = default;

    ArrayReferenceTuple(const Types*... data, size_t size)
        : m_size(size)
    {
        if (m_size)
            m_data = { data... };
    }

    bool isEmpty() const { return !m_size; }
    size_t size() const { return m_size; }

    template<unsigned I>
    auto data() const
    {
        return std::get<I>(m_data);
    }

    template<unsigned I>
    auto span() const
    {
        return unsafeMakeSpan(std::get<I>(m_data), m_size);
    }

private:
    size_t m_size { 0 };
    std::tuple<const Types*...> m_data;
};

}
