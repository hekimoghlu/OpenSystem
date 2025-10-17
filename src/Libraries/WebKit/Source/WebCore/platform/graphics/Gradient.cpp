/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "Gradient.h"

#include "FloatRect.h"
#include <wtf/HashFunctions.h>
#include <wtf/Hasher.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<Gradient> Gradient::create(Data&& data, ColorInterpolationMethod colorInterpolationMethod, GradientSpreadMethod spreadMethod, GradientColorStops&& stops, std::optional<RenderingResourceIdentifier> renderingResourceIdentifier)
{
    return adoptRef(*new Gradient(WTFMove(data), colorInterpolationMethod, spreadMethod, WTFMove(stops), renderingResourceIdentifier));
}

Gradient::Gradient(Data&& data, ColorInterpolationMethod colorInterpolationMethod, GradientSpreadMethod spreadMethod, GradientColorStops&& stops, std::optional<RenderingResourceIdentifier> renderingResourceIdentifier)
    : RenderingResource(renderingResourceIdentifier)
    , m_data { WTFMove(data) }
    , m_colorInterpolationMethod { colorInterpolationMethod }
    , m_spreadMethod { spreadMethod }
    , m_stops { WTFMove(stops) }
{
}

void Gradient::adjustParametersForTiledDrawing(FloatSize& size, FloatRect& srcRect, const FloatSize& spacing)
{
    if (srcRect.isEmpty())
        return;

    if (!spacing.isZero())
        return;

    WTF::switchOn(m_data,
        [&] (const LinearData& data) {
            if (data.point0.x() == data.point1.x()) {
                size.setWidth(1);
                srcRect.setWidth(1);
                srcRect.setX(0);
                return;
            }
            if (data.point0.y() != data.point1.y())
                return;

            size.setHeight(1);
            srcRect.setHeight(1);
            srcRect.setY(0);
        },
        [] (const RadialData&) {
        },
        [] (const ConicData&) {
        }
    );
}

bool Gradient::isZeroSize() const
{
    return WTF::switchOn(m_data,
        [] (const LinearData& data) {
            return data.point0.x() == data.point1.x() && data.point0.y() == data.point1.y();
        },
        [] (const RadialData& data) {
            return data.point0.x() == data.point1.x() && data.point0.y() == data.point1.y() && data.startRadius == data.endRadius;
        },
        [] (const ConicData&) {
            return false;
        }
    );
}

void Gradient::addColorStop(GradientColorStop&& stop)
{
    m_stops.addColorStop(WTFMove(stop));
    m_cachedHash = 0;
    stopsChanged();
}

static void add(Hasher& hasher, const Gradient::LinearData& data)
{
    add(hasher, data.point0, data.point1);
}

static void add(Hasher& hasher, const Gradient::RadialData& data)
{
    add(hasher, data.point0, data.point1, data.startRadius, data.endRadius, data.aspectRatio);
}

static void add(Hasher& hasher, const Gradient::ConicData& data)
{
    add(hasher, data.point0, data.angleRadians);
}

unsigned Gradient::hash() const
{
    if (!m_cachedHash)
        m_cachedHash = computeHash(m_data, m_colorInterpolationMethod, m_spreadMethod, m_stops.sorted());
    return m_cachedHash;
}

TextStream& operator<<(TextStream& ts, const Gradient& gradient)
{
    WTF::switchOn(gradient.data(),
        [&] (const Gradient::LinearData& data) {
            ts.dumpProperty("p0", data.point0);
            ts.dumpProperty("p1", data.point1);
        },
        [&] (const Gradient::RadialData& data) {
            ts.dumpProperty("p0", data.point0);
            ts.dumpProperty("p1", data.point1);
            ts.dumpProperty("start-radius", data.startRadius);
            ts.dumpProperty("end-radius", data.endRadius);
            ts.dumpProperty("aspect-ratio", data.aspectRatio);
        },
        [&] (const Gradient::ConicData& data) {
            ts.dumpProperty("p0", data.point0);
            ts.dumpProperty("angle-radians", data.angleRadians);
        }
    );
    ts.dumpProperty("color-interpolation-method", gradient.colorInterpolationMethod());
    ts.dumpProperty("spread-method", gradient.spreadMethod());
    ts.dumpProperty("stops", gradient.stops());
    return ts;
}

} // namespace WebCore
