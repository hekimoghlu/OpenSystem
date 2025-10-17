/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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

#include "AffineTransform.h"
#include "FloatPoint.h"
#include "FloatQuad.h"
#include "LayoutSize.h"
#include "TransformationMatrix.h"
#include <optional>

namespace WTF {
class TextStream;
}

namespace WebCore {

class TransformState {
public:
    enum TransformDirection { ApplyTransformDirection, UnapplyInverseTransformDirection };
    enum TransformAccumulation { FlattenTransform, AccumulateTransform };
    enum TransformMatrixTracking { DoNotTrackTransformMatrix, TrackSVGCTMMatrix, TrackSVGScreenCTMMatrix };

    TransformState(TransformDirection mappingDirection, const FloatPoint& p, const FloatQuad& quad)
        : m_lastPlanarPoint(p)
        , m_lastPlanarQuad(quad)
        , m_mapPoint(true)
        , m_mapQuad(true)
        , m_direction(mappingDirection)
    {
    }
    
    TransformState(TransformDirection mappingDirection, const FloatPoint& p)
        : m_lastPlanarPoint(p)
        , m_mapPoint(true)
        , m_mapQuad(false)
        , m_direction(mappingDirection)
    {
    }
    
    TransformState(TransformDirection mappingDirection, const FloatQuad& quad)
        : m_lastPlanarQuad(quad)
        , m_mapPoint(false)
        , m_mapQuad(true)
        , m_direction(mappingDirection)
    {
    }
    
    TransformState(const TransformState& other) { *this = other; }

    WEBCORE_EXPORT TransformState& operator=(const TransformState&);

    void setQuad(const FloatQuad& quad)
    {
        // We must be in a flattened state (no accumulated offset) when setting this quad.
        ASSERT(m_accumulatedOffset == LayoutSize());
        m_lastPlanarQuad = quad;
    }

    void setSecondaryQuad(const std::optional<FloatQuad>& quad)
    {
        // We must be in a flattened state (no accumulated offset) when setting this secondary quad.
        ASSERT(m_accumulatedOffset == LayoutSize());
        m_lastPlanarSecondaryQuad = quad;
    }

    void setLastPlanarSecondaryQuad(const std::optional<FloatQuad>&);

    void setTransformMatrixTracking(TransformMatrixTracking tracking) { m_tracking = tracking; }
    TransformMatrixTracking transformMatrixTracking() const { return m_tracking; }

    void move(LayoutUnit x, LayoutUnit y, TransformAccumulation accumulate = FlattenTransform)
    {
        move(LayoutSize(x, y), accumulate);
    }

    void move(const LayoutSize&, TransformAccumulation = FlattenTransform);
    void applyTransform(const AffineTransform& transformFromContainer, TransformAccumulation = FlattenTransform, bool* wasClamped = nullptr);
    WEBCORE_EXPORT void applyTransform(const TransformationMatrix& transformFromContainer, TransformAccumulation = FlattenTransform, bool* wasClamped = nullptr);
    WEBCORE_EXPORT void flatten(bool* wasClamped = nullptr);

    // Return the coords of the point or quad in the last flattened layer
    FloatPoint lastPlanarPoint() const { return m_lastPlanarPoint; }
    FloatQuad lastPlanarQuad() const { return m_lastPlanarQuad; }
    std::optional<FloatQuad> lastPlanarSecondaryQuad() const { return m_lastPlanarSecondaryQuad; }
    bool isMappingSecondaryQuad() const { return m_lastPlanarSecondaryQuad.has_value(); }

    // Return the point or quad mapped through the current transform
    FloatPoint mappedPoint(bool* wasClamped = nullptr) const;
    WEBCORE_EXPORT FloatQuad mappedQuad(bool* wasClamped = nullptr) const;
    WEBCORE_EXPORT std::optional<FloatQuad> mappedSecondaryQuad(bool* wasClamped = nullptr) const;

    TransformationMatrix* accumulatedTransform() const { return m_accumulatedTransform.get(); }
    std::unique_ptr<TransformationMatrix> releaseTrackedTransform() { return WTFMove(m_trackedTransform); }
    TransformDirection direction() const { return m_direction; }

private:
    void translateTransform(const LayoutSize&);
    void translateMappedCoordinates(const LayoutSize&);
    void flattenWithTransform(const TransformationMatrix&, bool* wasClamped);
    void applyAccumulatedOffset();

    bool shouldFlattenBefore(TransformAccumulation accumulate = FlattenTransform);
    bool shouldFlattenAfter(TransformAccumulation accumulate = FlattenTransform);
    
    TransformDirection inverseDirection() const;

    void mapQuad(FloatQuad&, TransformDirection, bool* clamped = nullptr) const;
    
    FloatPoint m_lastPlanarPoint;
    FloatQuad m_lastPlanarQuad;
    std::optional<FloatQuad> m_lastPlanarSecondaryQuad;

    // We only allocate the transform if we need to
    std::unique_ptr<TransformationMatrix> m_accumulatedTransform;
    std::unique_ptr<TransformationMatrix> m_trackedTransform;
    LayoutSize m_accumulatedOffset;
    bool m_accumulatingTransform { false };
    bool m_mapPoint;
    bool m_mapQuad;
    TransformMatrixTracking m_tracking { DoNotTrackTransformMatrix };
    TransformDirection m_direction;
};

inline TransformState::TransformDirection TransformState::inverseDirection() const
{
    return m_direction == ApplyTransformDirection ? UnapplyInverseTransformDirection : ApplyTransformDirection;
}

WTF::TextStream& operator<<(WTF::TextStream&, const TransformState&);

} // namespace WebCore
