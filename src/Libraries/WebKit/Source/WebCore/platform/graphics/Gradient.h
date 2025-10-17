/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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

#include "Color.h"
#include "ColorInterpolationMethod.h"
#include "FloatPoint.h"
#include "GradientColorStops.h"
#include "GraphicsTypes.h"
#include "RenderingResource.h"
#include <variant>
#include <wtf/Vector.h>

#if USE(SKIA)
#include <skia/core/SkShader.h>
#endif

#if USE(CG)
#include "GradientRendererCG.h"
#endif

#if USE(CG)
typedef struct CGContext* CGContextRef;
#endif

#if USE(CAIRO)
typedef struct _cairo_pattern cairo_pattern_t;
#endif

namespace WTF {
class TextStream;
}

namespace WebCore {

class AffineTransform;
class FloatRect;
class GraphicsContext;

class Gradient : public RenderingResource {
public:
    struct LinearData {
        FloatPoint point0;
        FloatPoint point1;
    };

    struct RadialData {
        FloatPoint point0;
        FloatPoint point1;
        float startRadius;
        float endRadius;
        float aspectRatio; // For elliptical gradient, width / height.
    };

    struct ConicData {
        FloatPoint point0;
        float angleRadians;
    };

    using Data = std::variant<LinearData, RadialData, ConicData>;

    WEBCORE_EXPORT static Ref<Gradient> create(Data&&, ColorInterpolationMethod, GradientSpreadMethod = GradientSpreadMethod::Pad, GradientColorStops&& = { }, std::optional<RenderingResourceIdentifier> = std::nullopt);

    const Data& data() const { return m_data; }
    ColorInterpolationMethod colorInterpolationMethod() const { return m_colorInterpolationMethod; }
    GradientSpreadMethod spreadMethod() const { return m_spreadMethod; }
    const GradientColorStops& stops() const { return m_stops; }

    WEBCORE_EXPORT void addColorStop(GradientColorStop&&);

    bool isZeroSize() const;

    void fill(GraphicsContext&, const FloatRect&);
    void adjustParametersForTiledDrawing(FloatSize&, FloatRect&, const FloatSize& spacing);

    unsigned hash() const;

#if USE(CAIRO)
    RefPtr<cairo_pattern_t> createPattern(float globalAlpha, const AffineTransform&);
#endif

#if USE(CG)
    void paint(GraphicsContext&);
    void paint(CGContextRef);
#endif

#if USE(SKIA)
    sk_sp<SkShader> shader(float globalAlpha, const AffineTransform&);
#endif

private:
    Gradient(Data&&, ColorInterpolationMethod, GradientSpreadMethod, GradientColorStops&&, std::optional<RenderingResourceIdentifier>);

    bool isGradient() const final { return true; }

    void stopsChanged();

    Data m_data;
    ColorInterpolationMethod m_colorInterpolationMethod;
    GradientSpreadMethod m_spreadMethod;
    GradientColorStops m_stops;
    mutable unsigned m_cachedHash { 0 };

#if USE(CG)
    std::optional<GradientRendererCG> m_platformRenderer;
#endif

#if USE(SKIA)
    sk_sp<SkShader> m_shader;
#endif

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const Gradient&);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Gradient)
    static bool isType(const WebCore::RenderingResource& renderingResource) { return renderingResource.isGradient(); }
SPECIALIZE_TYPE_TRAITS_END()
