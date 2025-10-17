/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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

#include <iterator>

namespace WTF {

template<typename Iterator>
class IteratorRange {
    WTF_MAKE_FAST_ALLOCATED;
public:
    using reverse_iterator = std::reverse_iterator<Iterator>;

    IteratorRange(Iterator begin, Iterator end)
        : m_begin(WTFMove(begin))
        , m_end(WTFMove(end))
    {
    }

    auto begin() const { return m_begin; }
    auto end() const { return m_end; }
    auto rbegin() const { return reverse_iterator { m_end }; }
    auto rend() const { return reverse_iterator { m_begin }; }

private:
    Iterator m_begin;
    Iterator m_end;
};

template<typename Iterator>
IteratorRange<Iterator> makeIteratorRange(Iterator&& begin, Iterator&& end)
{
    return IteratorRange<Iterator>(std::forward<Iterator>(begin), std::forward<Iterator>(end));
}

template<typename Container>
IteratorRange<typename Container::reverse_iterator> makeReversedRange(Container& container)
{
    return makeIteratorRange(std::rbegin(container), std::rend(container));
}

template<typename Container>
IteratorRange<typename Container::const_reverse_iterator> makeReversedRange(const Container& container)
{
    return makeIteratorRange(std::crbegin(container), std::crend(container));
}

template<typename Container, typename Iterator>
class SizedIteratorRange {
    WTF_MAKE_FAST_ALLOCATED;
public:
    SizedIteratorRange(const Container& container, Iterator begin, Iterator end)
        : m_container(container)
        , m_begin(WTFMove(begin))
        , m_end(WTFMove(end))
    {
    }

    auto size() const -> decltype(std::declval<Container>().size()) { return m_container.size(); }
    bool isEmpty() const { return m_container.isEmpty(); }
    Iterator begin() const { return m_begin; }
    Iterator end() const { return m_end; }

private:
    const Container& m_container;
    Iterator m_begin;
    Iterator m_end;
};

template<typename Container, typename Iterator>
SizedIteratorRange<Container, Iterator> makeSizedIteratorRange(const Container& container, Iterator&& begin, Iterator&& end)
{
    return SizedIteratorRange<Container, Iterator>(container, std::forward<Iterator>(begin), std::forward<Iterator>(end));
}

} // namespace WTF

using WTF::IteratorRange;
using WTF::makeReversedRange;
