/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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

#include <wtf/StdSet.h>

namespace WebCore {

class RenderBox;
class RenderObject;
    
class OrderIterator {
public:
    friend class OrderIteratorPopulator;

    explicit OrderIterator(RenderBox&);

    RenderBox* currentChild() const { return m_currentChild; }
    RenderBox* first();
    RenderBox* next();
    OrderIterator reverse();
    bool shouldSkipChild(const RenderObject&) const;

private:
    void reset();

    RenderBox& m_containerBox;
    RenderBox* m_currentChild;

    using OrderValues = StdSet<int>;
    OrderValues m_orderValues;
    OrderValues::const_iterator m_orderValuesIterator;
    bool m_isReset { false };
    bool m_reversedOrder { false };
};

class OrderIteratorPopulator {
public:
    explicit OrderIteratorPopulator(OrderIterator& iterator)
        : m_iterator(iterator)
    {
        m_iterator.m_orderValues.clear();
    }
    ~OrderIteratorPopulator();

    bool collectChild(const RenderBox&);

private:
    OrderIterator& m_iterator;
};

} // namespace WebCore
