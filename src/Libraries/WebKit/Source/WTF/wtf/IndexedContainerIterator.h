/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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

#include <type_traits>

namespace WTF {

template<class Container>
class IndexedContainerIterator {
    WTF_MAKE_FAST_ALLOCATED;
public:
    IndexedContainerIterator()
        : m_container(nullptr)
        , m_index(0)
    {
    }

    IndexedContainerIterator(const Container& container, unsigned index)
        : m_container(&container)
        , m_index(findNext(index))
    {
    }

    auto operator*() -> typename std::invoke_result<decltype(&Container::at), const Container, unsigned>::type
    {
        return m_container->at(m_index);
    }

    IndexedContainerIterator& operator++()
    {
        m_index = findNext(m_index + 1);
        return *this;
    }

    bool operator==(const IndexedContainerIterator& other) const
    {
        ASSERT(m_container == other.m_container);
        return m_index == other.m_index;
    }

private:
    unsigned findNext(unsigned index)
    {
        while (index < m_container->size() && !m_container->at(index))
            index++;
        return index;
    }

    const Container* m_container;
    unsigned m_index;
};

} // namespace WTF
