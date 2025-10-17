/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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
#include "PathSegment.h"

#include "PathElement.h"
#include "PathImpl.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

PathSegment::PathSegment(Data&& data)
    : m_data(WTFMove(data))
{
}

FloatPoint PathSegment::calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const
{
    return WTF::switchOn(m_data, [&](auto& data) {
        return data.calculateEndPoint(currentPoint, lastMoveToPoint);
    });
}

std::optional<FloatPoint> PathSegment::tryGetEndPointWithoutContext() const
{
    return WTF::switchOn(m_data, [&](auto& data) {
        return data.tryGetEndPointWithoutContext();
    });
}

void PathSegment::extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const
{
    WTF::switchOn(m_data, [&](auto& data) {
        data.extendFastBoundingRect(currentPoint, lastMoveToPoint, boundingRect);
    });
}

void PathSegment::extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const
{
    WTF::switchOn(m_data, [&](auto& data) {
        data.extendBoundingRect(currentPoint, lastMoveToPoint, boundingRect);
    });
}

bool PathSegment::canApplyElements() const
{
    return WTF::switchOn(m_data, [&](auto& data) {
        return data.canApplyElements;
    });
}

bool PathSegment::applyElements(const PathElementApplier& applier) const
{
    return WTF::switchOn(m_data, [&]<typename DataType>(DataType& data) -> bool {
        if constexpr (DataType::canApplyElements) {
            data.applyElements(applier);
            return true;
        }
        return false;
    });
}

bool PathSegment::canTransform() const
{
    return WTF::switchOn(m_data, [&](auto& data) {
        return data.canTransform;
    });
}

bool PathSegment::transform(const AffineTransform& transform)
{
    return WTF::switchOn(m_data, [&]<typename DataType>(DataType& data) {
        if constexpr (DataType::canTransform) {
            data.transform(transform);
            return true;
        }
        return false;
    });
}

TextStream& operator<<(TextStream& ts, const PathSegment& segment)
{
    return WTF::switchOn(segment.data(), [&](auto& data) -> TextStream& {
        return ts << data;
    });
}

} // namespace WebCore
