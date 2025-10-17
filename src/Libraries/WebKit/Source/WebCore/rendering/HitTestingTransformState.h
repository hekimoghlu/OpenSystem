/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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

#include "FloatPoint.h"
#include "FloatQuad.h"
#include "IntSize.h"
#include "TransformationMatrix.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

// FIXME: Now that TransformState lazily creates its TransformationMatrix it takes up less space.
// So there's really no need for a ref counted version. So This class should be removed and replaced
// with TransformState. There are some minor differences (like the way translate() works slightly
// differently than move()) so care has to be taken when this is done.
class HitTestingTransformState : public RefCounted<HitTestingTransformState> {
public:
    static Ref<HitTestingTransformState> create(const FloatPoint& p, const FloatQuad& quad, const FloatQuad& area)
    {
        return adoptRef(*new HitTestingTransformState(p, quad, area));
    }

    static Ref<HitTestingTransformState> create(const HitTestingTransformState& other)
    {
        return adoptRef(*new HitTestingTransformState(other));
    }

    enum TransformAccumulation { FlattenTransform, AccumulateTransform };
    void translate(int x, int y, TransformAccumulation);
    void applyTransform(const TransformationMatrix& transformFromContainer, TransformAccumulation);

    FloatPoint mappedPoint() const;
    FloatQuad mappedQuad() const;
    FloatQuad mappedArea() const;
    LayoutRect boundsOfMappedArea() const;
    void flatten();

    FloatPoint m_lastPlanarPoint;
    FloatQuad m_lastPlanarQuad;
    FloatQuad m_lastPlanarArea;
    TransformationMatrix m_accumulatedTransform;
    bool m_accumulatingTransform;

private:
    HitTestingTransformState(const FloatPoint& p, const FloatQuad& quad, const FloatQuad& area)
        : m_lastPlanarPoint(p)
        , m_lastPlanarQuad(quad)
        , m_lastPlanarArea(area)
        , m_accumulatingTransform(false)
    {
    }
    
    HitTestingTransformState(const HitTestingTransformState& other)
        : RefCounted<HitTestingTransformState>()
        , m_lastPlanarPoint(other.m_lastPlanarPoint)
        , m_lastPlanarQuad(other.m_lastPlanarQuad)
        , m_lastPlanarArea(other.m_lastPlanarArea)
        , m_accumulatedTransform(other.m_accumulatedTransform)
        , m_accumulatingTransform(other.m_accumulatingTransform)
    {
    }
    
    void flattenWithTransform(const TransformationMatrix&);
};

} // namespace WebCore
