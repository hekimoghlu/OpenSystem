/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include "GraphicsContextState.h"

#include <wtf/MathExtras.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

GraphicsContextState::GraphicsContextState(const ChangeFlags& changeFlags, InterpolationQuality imageInterpolationQuality)
    : m_changeFlags(changeFlags)
    , m_imageInterpolationQuality(imageInterpolationQuality)
{
}

void GraphicsContextState::repurpose(Purpose purpose)
{
    if (purpose == Purpose::Initial)
        m_changeFlags = { };

#if USE(CG)
    // CGContextBeginTransparencyLayer() sets the CG global alpha to 1. Keep the clone's alpha in sync.
    if (purpose == Purpose::TransparencyLayer)
        m_alpha = 1;
#endif

    m_purpose = purpose;
}

GraphicsContextState GraphicsContextState::clone(Purpose purpose) const
{
    auto clone = *this;
    clone.repurpose(purpose);
    return clone;
}

bool GraphicsContextState::containsOnlyInlineChanges() const
{
    if (m_changeFlags.isEmpty() || m_changeFlags != (m_changeFlags & basicChangeFlags))
        return false;

    if (m_changeFlags.contains(Change::StrokeBrush) && !m_strokeBrush.isInlineColor())
        return false;

    if (m_changeFlags.contains(Change::FillBrush) && !m_fillBrush.isInlineColor())
        return false;

    return true;
}

bool GraphicsContextState::containsOnlyInlineStrokeChanges() const
{
    if (m_changeFlags.isEmpty() || m_changeFlags != (m_changeFlags & strokeChangeFlags))
        return false;

    if (m_changeFlags.contains(Change::StrokeBrush) && !m_strokeBrush.isInlineColor())
        return false;

    return true;
}

void GraphicsContextState::mergeLastChanges(const GraphicsContextState& state, const std::optional<GraphicsContextState>& lastDrawingState)
{
    for (auto change : state.changes())
        mergeSingleChange(state, toIndex(change), lastDrawingState);
}

void GraphicsContextState::mergeSingleChange(const GraphicsContextState& state, ChangeIndex changeIndex, const std::optional<GraphicsContextState>& lastDrawingState)
{
    auto mergeChange = [&](auto GraphicsContextState::*property) {
        if (this->*property == state.*property)
            return;
        this->*property = state.*property;
        m_changeFlags.set(changeIndex.toChange(), !lastDrawingState || (*lastDrawingState).*property != this->*property);
    };

    switch (changeIndex.value) {
    case toIndex(Change::FillBrush).value:
        mergeChange(&GraphicsContextState::m_fillBrush);
        break;
    case toIndex(Change::FillRule).value:
        mergeChange(&GraphicsContextState::m_fillRule);
        break;

    case toIndex(Change::StrokeBrush).value:
        mergeChange(&GraphicsContextState::m_strokeBrush);
        break;
    case toIndex(Change::StrokeThickness).value:
        mergeChange(&GraphicsContextState::m_strokeThickness);
        break;
    case toIndex(Change::StrokeStyle).value:
        mergeChange(&GraphicsContextState::m_strokeStyle);
        break;

    case toIndex(Change::CompositeMode).value:
        mergeChange(&GraphicsContextState::m_compositeMode);
        break;
    case toIndex(Change::DropShadow).value:
        mergeChange(&GraphicsContextState::m_dropShadow);
        break;
    case toIndex(Change::Style).value:
        mergeChange(&GraphicsContextState::m_style);
        break;

    case toIndex(Change::Alpha).value:
        mergeChange(&GraphicsContextState::m_alpha);
        break;
    case toIndex(Change::TextDrawingMode).value:
        mergeChange(&GraphicsContextState::m_textDrawingMode);
        break;
    case toIndex(Change::ImageInterpolationQuality).value:
        mergeChange(&GraphicsContextState::m_imageInterpolationQuality);
        break;

    case toIndex(Change::ShouldAntialias).value:
        mergeChange(&GraphicsContextState::m_shouldAntialias);
        break;
    case toIndex(Change::ShouldSmoothFonts).value:
        mergeChange(&GraphicsContextState::m_shouldSmoothFonts);
        break;
    case toIndex(Change::ShouldSubpixelQuantizeFonts).value:
        mergeChange(&GraphicsContextState::m_shouldSubpixelQuantizeFonts);
        break;
    case toIndex(Change::ShadowsIgnoreTransforms).value:
        mergeChange(&GraphicsContextState::m_shadowsIgnoreTransforms);
        break;
    case toIndex(Change::DrawLuminanceMask).value:
        mergeChange(&GraphicsContextState::m_drawLuminanceMask);
        break;
    case toIndex(Change::UseDarkAppearance).value:
        mergeChange(&GraphicsContextState::m_useDarkAppearance);
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

void GraphicsContextState::mergeAllChanges(const GraphicsContextState& state)
{
    auto mergeChange = [&](Change change, auto GraphicsContextState::*property) {
        if (this->*property == state.*property)
            return;
        this->*property = state.*property;
        m_changeFlags.add(change);
    };

    mergeChange(Change::FillBrush,                   &GraphicsContextState::m_fillBrush);
    mergeChange(Change::FillRule,                    &GraphicsContextState::m_fillRule);

    mergeChange(Change::StrokeBrush,                 &GraphicsContextState::m_strokeBrush);
    mergeChange(Change::StrokeThickness,             &GraphicsContextState::m_strokeThickness);
    mergeChange(Change::StrokeStyle,                 &GraphicsContextState::m_strokeStyle);

    mergeChange(Change::CompositeMode,               &GraphicsContextState::m_compositeMode);
    mergeChange(Change::DropShadow,                  &GraphicsContextState::m_dropShadow);
    mergeChange(Change::Style,                       &GraphicsContextState::m_style);

    mergeChange(Change::Alpha,                       &GraphicsContextState::m_alpha);
    mergeChange(Change::ImageInterpolationQuality,   &GraphicsContextState::m_textDrawingMode);
    mergeChange(Change::TextDrawingMode,             &GraphicsContextState::m_imageInterpolationQuality);

    mergeChange(Change::ShouldAntialias,             &GraphicsContextState::m_shouldAntialias);
    mergeChange(Change::ShouldSmoothFonts,           &GraphicsContextState::m_shouldSmoothFonts);
    mergeChange(Change::ShouldSubpixelQuantizeFonts, &GraphicsContextState::m_shouldSubpixelQuantizeFonts);
    mergeChange(Change::ShadowsIgnoreTransforms,     &GraphicsContextState::m_shadowsIgnoreTransforms);
    mergeChange(Change::DrawLuminanceMask,           &GraphicsContextState::m_drawLuminanceMask);
    mergeChange(Change::UseDarkAppearance,           &GraphicsContextState::m_useDarkAppearance);
}

static ASCIILiteral stateChangeName(GraphicsContextState::Change change)
{
    switch (change) {
    case GraphicsContextState::Change::FillBrush:
        return "fill-brush"_s;

    case GraphicsContextState::Change::FillRule:
        return "fill-rule"_s;

    case GraphicsContextState::Change::StrokeBrush:
        return "stroke-brush"_s;

    case GraphicsContextState::Change::StrokeThickness:
        return "stroke-thickness"_s;

    case GraphicsContextState::Change::StrokeStyle:
        return "stroke-style"_s;

    case GraphicsContextState::Change::CompositeMode:
        return "composite-mode"_s;

    case GraphicsContextState::Change::DropShadow:
        return "drop-shadow"_s;

    case GraphicsContextState::Change::Style:
        return "style"_s;

    case GraphicsContextState::Change::Alpha:
        return "alpha"_s;

    case GraphicsContextState::Change::ImageInterpolationQuality:
        return "image-interpolation-quality"_s;

    case GraphicsContextState::Change::TextDrawingMode:
        return "text-drawing-mode"_s;

    case GraphicsContextState::Change::ShouldAntialias:
        return "should-antialias"_s;

    case GraphicsContextState::Change::ShouldSmoothFonts:
        return "should-smooth-fonts"_s;

    case GraphicsContextState::Change::ShouldSubpixelQuantizeFonts:
        return "should-subpixel-quantize-fonts"_s;

    case GraphicsContextState::Change::ShadowsIgnoreTransforms:
        return "shadows-ignore-transforms"_s;

    case GraphicsContextState::Change::DrawLuminanceMask:
        return "draw-luminance-mask"_s;

    case GraphicsContextState::Change::UseDarkAppearance:
        return "use-dark-appearance"_s;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

TextStream& GraphicsContextState::dump(TextStream& ts) const
{
    auto dump = [&](Change change, auto GraphicsContextState::*property) {
        if (m_changeFlags.contains(change))
            ts.dumpProperty(stateChangeName(change), this->*property);
    };

    ts.dumpProperty("change-flags", m_changeFlags);

    dump(Change::FillBrush,                     &GraphicsContextState::m_fillBrush);
    dump(Change::FillRule,                      &GraphicsContextState::m_fillRule);

    dump(Change::StrokeBrush,                   &GraphicsContextState::m_strokeBrush);
    dump(Change::StrokeThickness,               &GraphicsContextState::m_strokeThickness);
    dump(Change::StrokeStyle,                   &GraphicsContextState::m_strokeStyle);

    dump(Change::CompositeMode,                 &GraphicsContextState::m_compositeMode);
    dump(Change::DropShadow,                    &GraphicsContextState::m_dropShadow);
    dump(Change::Style,                         &GraphicsContextState::m_style);

    dump(Change::Alpha,                         &GraphicsContextState::m_alpha);
    dump(Change::ImageInterpolationQuality,     &GraphicsContextState::m_imageInterpolationQuality);
    dump(Change::TextDrawingMode,               &GraphicsContextState::m_textDrawingMode);

    dump(Change::ShouldAntialias,               &GraphicsContextState::m_shouldAntialias);
    dump(Change::ShouldSmoothFonts,             &GraphicsContextState::m_shouldSmoothFonts);
    dump(Change::ShouldSubpixelQuantizeFonts,   &GraphicsContextState::m_shouldSubpixelQuantizeFonts);
    dump(Change::ShadowsIgnoreTransforms,       &GraphicsContextState::m_shadowsIgnoreTransforms);
    dump(Change::DrawLuminanceMask,             &GraphicsContextState::m_drawLuminanceMask);
    dump(Change::UseDarkAppearance,             &GraphicsContextState::m_useDarkAppearance);
    return ts;
}

TextStream& operator<<(TextStream& ts, GraphicsContextState::Change change)
{
    ts << stateChangeName(change);
    return ts;
}

TextStream& operator<<(TextStream& ts, const GraphicsContextState& state)
{
    return state.dump(ts);
}

} // namespace WebCore
