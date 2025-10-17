/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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

#if USE(CAIRO)

#include "DashArray.h"
#include "GraphicsContext.h"
#include "GraphicsTypes.h"
#include <cairo.h>

namespace WebCore {

class AffineTransform;
class Color;
class FloatRect;
class FloatRoundedRect;
class FloatSize;
class Gradient;
class GraphicsContextState;
class Path;

namespace Cairo {

namespace State {

void setStrokeThickness(GraphicsContextCairo&, float);
void setStrokeStyle(GraphicsContextCairo&, StrokeStyle);

void setCompositeOperation(GraphicsContextCairo&, CompositeOperator, BlendMode);
void setShouldAntialias(GraphicsContextCairo&, bool);

void setCTM(GraphicsContextCairo&, const AffineTransform&);
AffineTransform getCTM(GraphicsContextCairo&);

IntRect getClipBounds(GraphicsContextCairo&);

bool isAcceleratedContext(GraphicsContextCairo&);

} // namespace State

enum class OrientationSizing {
    Normal,
    WidthAsHeight
};

struct FillSource {
    FillSource() = default;
    explicit FillSource(const GraphicsContextState&);
    FillSource(const GraphicsContextState&, Gradient&, const AffineTransform&);

    float globalAlpha { 0 };
    struct {
        RefPtr<cairo_pattern_t> object;
        FloatSize size;
        AffineTransform transform;
        bool repeatX { false };
        bool repeatY { false };
    } pattern;
    struct {
        RefPtr<cairo_pattern_t> base;
        RefPtr<cairo_pattern_t> alphaAdjusted;
    } gradient;
    Color color;

    WindRule fillRule { WindRule::NonZero };
};

struct StrokeSource {
    StrokeSource() = default;
    explicit StrokeSource(const GraphicsContextState&);

    float globalAlpha { 0 };
    RefPtr<cairo_pattern_t> pattern;
    struct {
        RefPtr<cairo_pattern_t> base;
        RefPtr<cairo_pattern_t> alphaAdjusted;
    } gradient;
    Color color;
};

struct ShadowState {
    ShadowState() = default;
    WEBCORE_EXPORT explicit ShadowState(const GraphicsContextState&);

    bool isVisible() const;
    bool isRequired(GraphicsContextCairo&) const;

    FloatSize offset;
    float blur { 0 };
    Color color;
    bool ignoreTransforms { false };

    float globalAlpha { 1.0 };
    CompositeOperator globalCompositeOperator { CompositeOperator::SourceOver };
};

void setLineCap(GraphicsContextCairo&, LineCap);
void setLineDash(GraphicsContextCairo&, const DashArray&, float);
void setLineJoin(GraphicsContextCairo&, LineJoin);
void setMiterLimit(GraphicsContextCairo&, float);

void fillRect(GraphicsContextCairo&, const FloatRect&, const FillSource&, const ShadowState&);
void fillRect(GraphicsContextCairo&, const FloatRect&, const Color&, const ShadowState&);
void fillRect(GraphicsContextCairo&, const FloatRect&, cairo_pattern_t*);
void fillRoundedRect(GraphicsContextCairo&, const FloatRoundedRect&, const Color&, const ShadowState&);
void fillRectWithRoundedHole(GraphicsContextCairo&, const FloatRect&, const FloatRoundedRect&, const FillSource&, const ShadowState&);
void fillPath(GraphicsContextCairo&, const Path&, const FillSource&, const ShadowState&);
void strokeRect(GraphicsContextCairo&, const FloatRect&, float, const StrokeSource&, const ShadowState&);
void strokePath(GraphicsContextCairo&, const Path&, const StrokeSource&, const ShadowState&);
void clearRect(GraphicsContextCairo&, const FloatRect&);

void drawGlyphs(GraphicsContextCairo&, const FillSource&, const StrokeSource&, const ShadowState&, const FloatPoint&, cairo_scaled_font_t*, double, const Vector<cairo_glyph_t>&, float, TextDrawingModeFlags, float, std::optional<GraphicsDropShadow>, FontSmoothingMode);

void drawPlatformImage(GraphicsContextCairo&, cairo_surface_t*, const FloatRect&, const FloatRect&, ImagePaintingOptions, float, const ShadowState&);
void drawPattern(GraphicsContextCairo&, cairo_surface_t*, const IntSize&, const FloatRect&, const FloatRect&, const AffineTransform&, const FloatPoint&, const FloatSize&, ImagePaintingOptions);
WEBCORE_EXPORT void drawSurface(GraphicsContextCairo&, cairo_surface_t*, const FloatRect&, const FloatRect&, InterpolationQuality, float, const ShadowState&, OrientationSizing operationSizing = OrientationSizing::Normal);

void drawRect(GraphicsContextCairo&, const FloatRect&, float, const Color&, StrokeStyle, const Color&);
void drawLine(GraphicsContextCairo&, const FloatPoint&, const FloatPoint&, StrokeStyle, const Color&, float, bool);
void drawLinesForText(GraphicsContextCairo&, const FloatPoint&, float thickness, const DashArray&, bool, bool, const Color&);
void drawDotsForDocumentMarker(GraphicsContextCairo&, const FloatRect&, DocumentMarkerLineStyle);
void drawEllipse(GraphicsContextCairo&, const FloatRect&, const Color&, StrokeStyle, const Color&, float);

void drawFocusRing(GraphicsContextCairo&, const Path&, float, const Color&);
void drawFocusRing(GraphicsContextCairo&, const Vector<FloatRect>&, float, const Color&);

void translate(GraphicsContextCairo&, float, float);
void rotate(GraphicsContextCairo&, float);
void scale(GraphicsContextCairo&, const FloatSize&);
void concatCTM(GraphicsContextCairo&, const AffineTransform&);

void beginTransparencyLayer(GraphicsContextCairo&, float);
void endTransparencyLayer(GraphicsContextCairo&);

void clip(GraphicsContextCairo&, const FloatRect&);
void clipOut(GraphicsContextCairo&, const FloatRect&);
void clipOut(GraphicsContextCairo&, const Path&);
void clipPath(GraphicsContextCairo&, const Path&, WindRule);

void clipToImageBuffer(GraphicsContextCairo&, cairo_surface_t*, const FloatRect&);

} // namespace Cairo
} // namespace WebCore

#endif // USE(CAIRO)
