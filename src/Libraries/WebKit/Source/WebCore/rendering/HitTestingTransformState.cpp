/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
#include "HitTestingTransformState.h"

#include "LayoutRect.h"

namespace WebCore {

void HitTestingTransformState::translate(int x, int y, TransformAccumulation accumulate)
{
    m_accumulatedTransform.translate(x, y);    
    if (accumulate == FlattenTransform)
        flattenWithTransform(m_accumulatedTransform);

    m_accumulatingTransform = accumulate == AccumulateTransform;
}

void HitTestingTransformState::applyTransform(const TransformationMatrix& transformFromContainer, TransformAccumulation accumulate)
{
    m_accumulatedTransform.multiply(transformFromContainer);
    if (accumulate == FlattenTransform)
        flattenWithTransform(m_accumulatedTransform);

    m_accumulatingTransform = accumulate == AccumulateTransform;
}

void HitTestingTransformState::flatten()
{
    flattenWithTransform(m_accumulatedTransform);
}

void HitTestingTransformState::flattenWithTransform(const TransformationMatrix& t)
{
    if (std::optional<TransformationMatrix> inverse = t.inverse()) {
        m_lastPlanarPoint = inverse.value().projectPoint(m_lastPlanarPoint);
        m_lastPlanarQuad = inverse.value().projectQuad(m_lastPlanarQuad);
        m_lastPlanarArea = inverse.value().projectQuad(m_lastPlanarArea);
    }

    m_accumulatedTransform.makeIdentity();
    m_accumulatingTransform = false;
}

FloatPoint HitTestingTransformState::mappedPoint() const
{
    if (auto inverse = m_accumulatedTransform.inverse())
        return inverse.value().projectPoint(m_lastPlanarPoint);
    return m_lastPlanarPoint;
}

FloatQuad HitTestingTransformState::mappedQuad() const
{
    if (auto inverse = m_accumulatedTransform.inverse())
        return inverse.value().projectQuad(m_lastPlanarQuad);
    return m_lastPlanarQuad;
}

FloatQuad HitTestingTransformState::mappedArea() const
{
    if (auto inverse = m_accumulatedTransform.inverse())
        return inverse.value().projectQuad(m_lastPlanarArea);
    return m_lastPlanarArea;
}

LayoutRect HitTestingTransformState::boundsOfMappedArea() const
{
    if (auto inverse = m_accumulatedTransform.inverse())
        return inverse.value().clampedBoundsOfProjectedQuad(m_lastPlanarArea);
    TransformationMatrix identity;
    return identity.clampedBoundsOfProjectedQuad(m_lastPlanarArea);
}

} // namespace WebCore
