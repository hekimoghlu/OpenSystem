/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
#include "FEOffset.h"

#include "Filter.h"
#include "FEOffsetSoftwareApplier.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FEOffset> FEOffset::create(float dx, float dy, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FEOffset(dx, dy, colorSpace));
}

FEOffset::FEOffset(float dx, float dy, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FEOffset, colorSpace)
    , m_dx(dx)
    , m_dy(dy)
{
}

bool FEOffset::operator==(const FEOffset& other) const
{
    return FilterEffect::operator==(other)
        && m_dx == other.m_dx
        && m_dy == other.m_dy;
}

bool FEOffset::setDx(float dx)
{
    if (m_dx == dx)
        return false;
    m_dx = dx;
    return true;
}

bool FEOffset::setDy(float dy)
{
    if (m_dy == dy)
        return false;
    m_dy = dy;
    return true;
}

FloatRect FEOffset::calculateImageRect(const Filter& filter, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const
{
    auto imageRect = inputImageRects[0];
    imageRect.move(filter.resolvedSize({ m_dx, m_dy }));
    return filter.clipToMaxEffectRect(imageRect, primitiveSubregion);
}

IntOutsets FEOffset::calculateOutsets(const FloatSize& offset)
{
    auto adjustedOffset = expandedIntSize(offset);

    IntOutsets outsets;
    if (adjustedOffset.height() < 0)
        outsets.setTop(-adjustedOffset.height());
    else
        outsets.setBottom(adjustedOffset.height());
    if (adjustedOffset.width() < 0)
        outsets.setLeft(-adjustedOffset.width());
    else
        outsets.setRight(adjustedOffset.width());

    return outsets;
}

bool FEOffset::resultIsAlphaImage(const FilterImageVector& inputs) const
{
    return inputs[0]->isAlphaImage();
}

std::unique_ptr<FilterEffectApplier> FEOffset::createSoftwareApplier() const
{
    return FilterEffectApplier::create<FEOffsetSoftwareApplier>(*this);
}

TextStream& FEOffset::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feOffset";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " dx=\"" << dx() << "\" dy=\"" << dy() << "\"";

    ts << "]\n";
    return ts;
}

} // namespace WebCore
