/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#include "FEFlood.h"

#include "ColorSerialization.h"
#include "FEFloodSoftwareApplier.h"
#include "Filter.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FEFlood> FEFlood::create(const Color& floodColor, float floodOpacity, DestinationColorSpace colorSpace)
{
#if USE(CG) || USE(SKIA)
    return adoptRef(*new FEFlood(floodColor, floodOpacity, colorSpace));
#else
    UNUSED_PARAM(colorSpace);
    return adoptRef(*new FEFlood(floodColor, floodOpacity));
#endif
}

FEFlood::FEFlood(const Color& floodColor, float floodOpacity, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FEFlood, colorSpace)
    , m_floodColor(floodColor)
    , m_floodOpacity(floodOpacity)
{
}

bool FEFlood::operator==(const FEFlood& other) const
{
    return FilterEffect::operator==(other)
        && m_floodColor == other.m_floodColor
        && m_floodOpacity == other.m_floodOpacity;
}

bool FEFlood::setFloodColor(const Color& color)
{
    if (m_floodColor == color)
        return false;
    m_floodColor = color;
    return true;
}

bool FEFlood::setFloodOpacity(float floodOpacity)
{
    if (m_floodOpacity == floodOpacity)
        return false;
    m_floodOpacity = floodOpacity;
    return true;
}

FloatRect FEFlood::calculateImageRect(const Filter& filter, std::span<const FloatRect>, const FloatRect& primitiveSubregion) const
{
    return filter.maxEffectRect(primitiveSubregion);
}

std::unique_ptr<FilterEffectApplier> FEFlood::createSoftwareApplier() const
{
    return FilterEffectApplier::create<FEFloodSoftwareApplier>(*this);
}

TextStream& FEFlood::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feFlood";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " flood-color=\"" << serializationForRenderTreeAsText(floodColor()) << "\"";
    ts << " flood-opacity=\"" << floodOpacity() << "\"";

    ts << "]\n";
    return ts;
}

} // namespace WebCore
