/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include "config.h"
#include "OrderIterator.h"

#include "RenderBox.h"
#include "RenderStyleInlines.h"

namespace WebCore {

OrderIterator::OrderIterator(RenderBox& containerBox)
    : m_containerBox(containerBox)
{
    reset();
}

RenderBox* OrderIterator::first()
{
    reset();
    return next();
}

RenderBox* OrderIterator::next()
{
    do {
        if (!m_currentChild) {
            if (m_orderValuesIterator == m_orderValues.end())
                return nullptr;
            
            if (!m_isReset) {
                ++m_orderValuesIterator;
                if (m_orderValuesIterator == m_orderValues.end())
                    return nullptr;
            } else
                m_isReset = false;
            m_currentChild = m_reversedOrder ? m_containerBox.lastChildBox() : m_containerBox.firstChildBox();
        } else {
            m_currentChild = m_reversedOrder ? m_currentChild->previousSiblingBox() : m_currentChild->nextSiblingBox();
        }
    } while (!m_currentChild || m_currentChild->style().order() != *m_orderValuesIterator);
    
    return m_currentChild;
}

void OrderIterator::reset()
{
    m_currentChild = nullptr;
    m_orderValuesIterator = m_orderValues.begin();
    m_isReset = true;
}

bool OrderIterator::shouldSkipChild(const RenderObject& child) const
{
    return child.isOutOfFlowPositioned() || child.isExcludedFromNormalLayout();
}

OrderIterator OrderIterator::reverse()
{
    OrderIterator reversedItr(*this);
    OrderValues reversedValues;

    for (auto valuesItr = m_orderValues.rbegin(); valuesItr != m_orderValues.rend(); valuesItr++)
        reversedValues.insert(*valuesItr);
    reversedItr.m_orderValues = reversedValues;
    reversedItr.m_reversedOrder = !m_reversedOrder;
    reversedItr.reset();

    return reversedItr;
}

OrderIteratorPopulator::~OrderIteratorPopulator()
{
    m_iterator.reset();
}

bool OrderIteratorPopulator::collectChild(const RenderBox& child)
{
    m_iterator.m_orderValues.insert(child.style().order());
    return !m_iterator.shouldSkipChild(child);
}


} // namespace WebCore
