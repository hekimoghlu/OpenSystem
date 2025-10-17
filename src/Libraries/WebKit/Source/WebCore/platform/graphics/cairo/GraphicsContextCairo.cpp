/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#include "GraphicsContextCairo.h"

#if USE(CAIRO)

#include "AffineTransform.h"
#include "CairoOperations.h"
#include "DecomposedGlyphs.h"
#include "FloatRect.h"
#include "FloatRoundedRect.h"
#include "Gradient.h"
#include "ImageBuffer.h"
#include "IntRect.h"
#include "NotImplemented.h"
#include "RefPtrCairo.h"

#if PLATFORM(WIN)
#include <cairo-win32.h>
#endif

#if USE(THEME_ADWAITA)
#include "Adwaita.h"
#endif

namespace WebCore {

// Encapsulates the additional painting state information we store for each
// pushed graphics state.
class GraphicsContextCairo::CairoState {
public:
    CairoState() = default;

    struct {
        RefPtr<cairo_pattern_t> pattern;
        cairo_matrix_t matrix;
    } m_mask;
};

GraphicsContextCairo::GraphicsContextCairo(RefPtr<cairo_t>&& context)
    : m_cr(WTFMove(context))
{
    m_cairoStateStack.append(CairoState());
    m_cairoState = &m_cairoStateStack.last();
}

GraphicsContextCairo::GraphicsContextCairo(cairo_surface_t* surface)
    : GraphicsContextCairo(adoptRef(cairo_create(surface)))
{
}

GraphicsContextCairo::~GraphicsContextCairo() = default;

bool GraphicsContextCairo::hasPlatformContext() const
{
    return true;
}

AffineTransform GraphicsContextCairo::getCTM(IncludeDeviceScale includeScale) const
{
    UNUSED_PARAM(includeScale);
    return Cairo::State::getCTM(*platformContext());
}

GraphicsContextCairo* GraphicsContextCairo::platformContext() const
{
    return const_cast<GraphicsContextCairo*>(this);
}

void GraphicsContextCairo::save(GraphicsContextState::Purpose purpose)
{
    GraphicsContext::save(purpose);

    m_cairoStateStack.append(CairoState());
    m_cairoState = &m_cairoStateStack.last();

    cairo_save(m_cr.get());
}

void GraphicsContextCairo::restore(GraphicsContextState::Purpose purpose)
{
    if (!stackSize())
        return;

    GraphicsContext::restore(purpose);

    if (m_cairoStateStack.isEmpty())
        return;

    if (m_cairoState->m_mask.pattern) {
        cairo_pop_group_to_source(m_cr.get());

        cairo_matrix_t matrix;
        cairo_get_matrix(m_cr.get(), &matrix);
        cairo_set_matrix(m_cr.get(), &m_cairoState->m_mask.matrix);
        cairo_mask(m_cr.get(), m_cairoState->m_mask.pattern.get());
        cairo_set_matrix(m_cr.get(), &matrix);
    }

    m_cairoStateStack.removeLast();
    ASSERT(!m_cairoStateStack.isEmpty());
    m_cairoState = &m_cairoStateStack.last();

    cairo_restore(m_cr.get());
}

// Draws a filled rectangle with a stroked border.
void GraphicsContextCairo::drawRect(const FloatRect& rect, float borderThickness)
{
    ASSERT(!rect.isEmpty());
    Cairo::drawRect(*this, rect, borderThickness, fillColor(), strokeStyle(), strokeColor());
}

void GraphicsContextCairo::drawNativeImageInternal(NativeImage& nativeImage, const FloatRect& destRect, const FloatRect& srcRect, ImagePaintingOptions options)
{
    auto& state = this->state();
    Cairo::drawPlatformImage(*this, nativeImage.platformImage().get(), destRect, srcRect, { options, state.imageInterpolationQuality() }, state.alpha(), Cairo::ShadowState(state));
}

// This is only used to draw borders, so we should not draw shadows.
void GraphicsContextCairo::drawLine(const FloatPoint& point1, const FloatPoint& point2)
{
    if (strokeStyle() == StrokeStyle::NoStroke)
        return;

    Cairo::drawLine(*this, point1, point2, strokeStyle(), strokeColor(), strokeThickness(), shouldAntialias());
}

// This method is only used to draw the little circles used in lists.
void GraphicsContextCairo::drawEllipse(const FloatRect& rect)
{
    Cairo::drawEllipse(*this, rect, fillColor(), strokeStyle(), strokeColor(), strokeThickness());
}

void GraphicsContextCairo::fillPath(const Path& path)
{
    if (path.isEmpty())
        return;

    auto& state = this->state();
    Cairo::fillPath(*this, path, Cairo::FillSource(state), Cairo::ShadowState(state));
}

void GraphicsContextCairo::strokePath(const Path& path)
{
    if (path.isEmpty())
        return;

    auto& state = this->state();
    Cairo::strokePath(*this, path, Cairo::StrokeSource(state), Cairo::ShadowState(state));
}

void GraphicsContextCairo::fillRect(const FloatRect& rect, RequiresClipToRect)
{
    auto& state = this->state();
    Cairo::fillRect(*this, rect, Cairo::FillSource(state), Cairo::ShadowState(state));
}

void GraphicsContextCairo::fillRect(const FloatRect& rect, Gradient& gradient, const AffineTransform& gradientSpaceTransform, RequiresClipToRect)
{
    auto& state = this->state();
    Cairo::fillRect(*this, rect, Cairo::FillSource(state, gradient, gradientSpaceTransform), Cairo::ShadowState(state));
}

void GraphicsContextCairo::fillRect(const FloatRect& rect, const Color& color)
{
    Cairo::fillRect(*this, rect, color, Cairo::ShadowState(state()));
}

void GraphicsContextCairo::resetClip()
{
    ASSERT_NOT_REACHED("resetClip is not supported on Cairo");
}

void GraphicsContextCairo::clip(const FloatRect& rect)
{
    Cairo::clip(*this, rect);
}

void GraphicsContextCairo::clipPath(const Path& path, WindRule clipRule)
{
    Cairo::clipPath(*this, path, clipRule);
}

IntRect GraphicsContextCairo::clipBounds() const
{
    return Cairo::State::getClipBounds(*platformContext());
}

void GraphicsContextCairo::clipToImageBuffer(ImageBuffer& buffer, const FloatRect& destRect)
{
    if (auto nativeImage = nativeImageForDrawing(buffer))
        Cairo::clipToImageBuffer(*this, nativeImage->platformImage().get(), destRect);
}

void GraphicsContextCairo::drawFocusRing(const Path& path, float outlineWidth, const Color& color)
{
#if USE(THEME_ADWAITA)
    Adwaita::paintFocus(*this, path, color);
    UNUSED_PARAM(outlineWidth);
    return;
#else
    Cairo::drawFocusRing(*this, path, outlineWidth, color);
#endif
}

void GraphicsContextCairo::drawFocusRing(const Vector<FloatRect>& rects, float outlineOffset, float outlineWidth, const Color& color)
{
#if USE(THEME_ADWAITA)
    Adwaita::paintFocus(*this, rects, color);
    UNUSED_PARAM(outlineOffset);
    UNUSED_PARAM(outlineWidth);
    return;
#else
    UNUSED_PARAM(outlineOffset);
    Cairo::drawFocusRing(*this, rects, outlineWidth, color);
#endif
}

void GraphicsContextCairo::drawLinesForText(const FloatPoint& point, float thickness, const DashArray& widths, bool printing, bool doubleUnderlines, StrokeStyle)
{
    if (widths.isEmpty())
        return;
    Cairo::drawLinesForText(*this, point, thickness, widths, printing, doubleUnderlines, strokeColor());
}

void GraphicsContextCairo::drawDotsForDocumentMarker(const FloatRect& rect, DocumentMarkerLineStyle style)
{
    Cairo::drawDotsForDocumentMarker(*this, rect, style);
}

void GraphicsContextCairo::translate(float x, float y)
{
    Cairo::translate(*this, x, y);
}

void GraphicsContextCairo::didUpdateState(GraphicsContextState& state)
{
    if (state.changes().contains(GraphicsContextState::Change::StrokeThickness))
        Cairo::State::setStrokeThickness(*this, state.strokeThickness());

    if (state.changes().contains(GraphicsContextState::Change::StrokeStyle))
        Cairo::State::setStrokeStyle(*this, state.strokeStyle());

    if (state.changes().contains(GraphicsContextState::Change::CompositeMode))
        Cairo::State::setCompositeOperation(*this, state.compositeMode().operation, state.compositeMode().blendMode);

    if (state.changes().contains(GraphicsContextState::Change::ShouldAntialias))
        Cairo::State::setShouldAntialias(*this, state.shouldAntialias());

    state.didApplyChanges();
}

void GraphicsContextCairo::concatCTM(const AffineTransform& transform)
{
    Cairo::concatCTM(*this, transform);
}

void GraphicsContextCairo::setCTM(const AffineTransform& transform)
{
    Cairo::State::setCTM(*this, transform);
}

void GraphicsContextCairo::beginTransparencyLayer(float opacity)
{
    GraphicsContext::beginTransparencyLayer(opacity);
    Cairo::beginTransparencyLayer(*this, opacity);
}

void GraphicsContextCairo::endTransparencyLayer()
{
    GraphicsContext::endTransparencyLayer();
    Cairo::endTransparencyLayer(*this);
}

void GraphicsContextCairo::clearRect(const FloatRect& rect)
{
    Cairo::clearRect(*this, rect);
}

void GraphicsContextCairo::strokeRect(const FloatRect& rect, float lineWidth)
{
    auto& state = this->state();
    Cairo::strokeRect(*this, rect, lineWidth, Cairo::StrokeSource(state), Cairo::ShadowState(state));
}

void GraphicsContextCairo::setLineCap(LineCap lineCap)
{
    Cairo::setLineCap(*this, lineCap);
}

void GraphicsContextCairo::setLineDash(const DashArray& dashes, float dashOffset)
{
    Cairo::setLineDash(*this, dashes, dashOffset);
}

void GraphicsContextCairo::setLineJoin(LineJoin lineJoin)
{
    Cairo::setLineJoin(*this, lineJoin);
}

void GraphicsContextCairo::setMiterLimit(float miter)
{
    Cairo::setMiterLimit(*this, miter);
}

void GraphicsContextCairo::clipOut(const Path& path)
{
    Cairo::clipOut(*this, path);
}

void GraphicsContextCairo::rotate(float radians)
{
    Cairo::rotate(*this, radians);
}

void GraphicsContextCairo::scale(const FloatSize& size)
{
    Cairo::scale(*this, size);
}

void GraphicsContextCairo::clipOut(const FloatRect& rect)
{
    Cairo::clipOut(*this, rect);
}

void GraphicsContextCairo::fillRoundedRectImpl(const FloatRoundedRect& rect, const Color& color)
{
    Cairo::fillRoundedRect(*this, rect, color, Cairo::ShadowState(state()));
}

void GraphicsContextCairo::fillRectWithRoundedHole(const FloatRect& rect, const FloatRoundedRect& roundedHoleRect, const Color& color)
{
    if (!color.isValid())
        return;

    auto& state = this->state();
    Cairo::fillRectWithRoundedHole(*this, rect, roundedHoleRect, Cairo::FillSource(state), Cairo::ShadowState(state));
}

void GraphicsContextCairo::drawPattern(NativeImage& nativeImage, const FloatRect& destRect, const FloatRect& tileRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    if (!patternTransform.isInvertible())
        return;

    Cairo::drawPattern(*this, nativeImage.platformImage().get(), nativeImage.size(), destRect, tileRect, patternTransform, phase, spacing, options);
}

RenderingMode GraphicsContextCairo::renderingMode() const
{
    return Cairo::State::isAcceleratedContext(*platformContext()) ? RenderingMode::Accelerated : RenderingMode::Unaccelerated;
}

cairo_t* GraphicsContextCairo::cr() const
{
    return m_cr.get();
}

Vector<float>& GraphicsContextCairo::layers()
{
    return m_layers;
}

void GraphicsContextCairo::pushImageMask(cairo_surface_t* surface, const FloatRect& rect)
{
    // We must call savePlatformState at least once before we can use image masking,
    // since we actually apply the mask in restorePlatformState.
    ASSERT(!m_cairoStateStack.isEmpty());
    m_cairoState->m_mask.pattern = adoptRef(cairo_pattern_create_for_surface(surface));
    cairo_get_matrix(m_cr.get(), &m_cairoState->m_mask.matrix);

    cairo_matrix_t matrix;
    cairo_matrix_init_translate(&matrix, -rect.x(), -rect.y());
    cairo_pattern_set_matrix(m_cairoState->m_mask.pattern.get(), &matrix);

    cairo_push_group(m_cr.get());
}

} // namespace WebCore

#endif // USE(CAIRO)
