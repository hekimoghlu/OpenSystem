/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#include "TextPaintStyle.h"

#include "ColorLuminance.h"
#include "FocusController.h"
#include "GraphicsContext.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PaintInfo.h"
#include "RenderStyleInlines.h"
#include "RenderText.h"
#include "RenderTheme.h"
#include "RenderView.h"
#include "Settings.h"

namespace WebCore {

TextPaintStyle::TextPaintStyle(const Color& color)
    : fillColor(color)
    , strokeColor(color)
{
}

bool TextPaintStyle::operator==(const TextPaintStyle& other) const
{
    return fillColor == other.fillColor && strokeColor == other.strokeColor && emphasisMarkColor == other.emphasisMarkColor
        && strokeWidth == other.strokeWidth && paintOrder == other.paintOrder && lineJoin == other.lineJoin
        && useDarkAppearance == other.useDarkAppearance
        && lineCap == other.lineCap && miterLimit == other.miterLimit;
}

bool textColorIsLegibleAgainstBackgroundColor(const Color& textColor, const Color& backgroundColor)
{
    // Uses the WCAG 2.0 definition of legibility: a contrast ratio of 4.5:1 or greater.
    // https://www.w3.org/TR/WCAG20/#visual-audio-contrast-contrast
    return contrastRatio(textColor, backgroundColor) >= 4.5;
}

static Color adjustColorForVisibilityOnBackground(const Color& textColor, const Color& backgroundColor)
{
    if (textColorIsLegibleAgainstBackgroundColor(textColor, backgroundColor))
        return textColor;

    if (textColor.luminance() > 0.5)
        return textColor.darkened();
    return textColor.lightened();
}

TextPaintStyle computeTextPaintStyle(const RenderText& renderer, const RenderStyle& lineStyle, const PaintInfo& paintInfo)
{
    auto& frame = renderer.frame();
    TextPaintStyle paintStyle;
    paintStyle.useDarkAppearance = frame.document() ? frame.document()->useDarkAppearance(&lineStyle) : false;

    auto viewportSize = frame.view() ? frame.view()->size() : IntSize();
    paintStyle.strokeWidth = lineStyle.computedStrokeWidth(viewportSize);
    paintStyle.paintOrder = lineStyle.paintOrder();
    paintStyle.lineJoin = lineStyle.joinStyle();
    paintStyle.lineCap = lineStyle.capStyle();
    paintStyle.miterLimit = lineStyle.strokeMiterLimit();
    
    if (paintInfo.forceTextColor()) {
        paintStyle.fillColor = paintInfo.forcedTextColor();
        paintStyle.strokeColor = paintInfo.forcedTextColor();
        paintStyle.emphasisMarkColor = paintInfo.forcedTextColor();
        return paintStyle;
    }

    if (lineStyle.insideDefaultButton()) {
        Page* page = renderer.frame().page();
        if (page && page->focusController().isActive()) {
            OptionSet<StyleColorOptions> options;
            if (page->settings().useSystemAppearance())
                options.add(StyleColorOptions::UseSystemAppearance);
            paintStyle.fillColor = RenderTheme::singleton().defaultButtonTextColor(options);
            return paintStyle;
        }
    }

    paintStyle.fillColor = lineStyle.visitedDependentColorWithColorFilter(CSSPropertyWebkitTextFillColor, paintInfo.paintBehavior);

    bool forceBackgroundToWhite = false;
    if (frame.document() && frame.document()->printing()) {
        if (lineStyle.printColorAdjust() == PrintColorAdjust::Economy)
            forceBackgroundToWhite = true;

        if (frame.settings().shouldPrintBackgrounds())
            forceBackgroundToWhite = false;

        if (forceBackgroundToWhite) {
            if (renderer.style().hasAnyBackgroundClipText())
                paintStyle.fillColor = Color::black;
        }
    }

    // Make the text fill color legible against a white background
    if (forceBackgroundToWhite)
        paintStyle.fillColor = adjustColorForVisibilityOnBackground(paintStyle.fillColor, Color::white);

    paintStyle.strokeColor = lineStyle.colorByApplyingColorFilter(lineStyle.computedStrokeColor());

    // Make the text stroke color legible against a white background
    if (forceBackgroundToWhite)
        paintStyle.strokeColor = adjustColorForVisibilityOnBackground(paintStyle.strokeColor, Color::white);

    paintStyle.emphasisMarkColor = lineStyle.visitedDependentColorWithColorFilter(CSSPropertyTextEmphasisColor);

    // Make the text stroke color legible against a white background
    if (forceBackgroundToWhite)
        paintStyle.emphasisMarkColor = adjustColorForVisibilityOnBackground(paintStyle.emphasisMarkColor, Color::white);

    return paintStyle;
}

TextPaintStyle computeTextSelectionPaintStyle(const TextPaintStyle& textPaintStyle, const RenderText& renderer, const RenderStyle& lineStyle, const PaintInfo& paintInfo, std::optional<ShadowData>& selectionShadow)
{
    TextPaintStyle selectionPaintStyle = textPaintStyle;

#if ENABLE(TEXT_SELECTION)
    Color foreground = paintInfo.forceTextColor() ? paintInfo.forcedTextColor() : renderer.selectionForegroundColor();
    if (foreground.isValid() && foreground != selectionPaintStyle.fillColor)
        selectionPaintStyle.fillColor = foreground;

    Color emphasisMarkForeground = paintInfo.forceTextColor() ? paintInfo.forcedTextColor() : renderer.selectionEmphasisMarkColor();
    if (emphasisMarkForeground.isValid() && emphasisMarkForeground != selectionPaintStyle.emphasisMarkColor)
        selectionPaintStyle.emphasisMarkColor = emphasisMarkForeground;

    if (auto pseudoStyle = renderer.selectionPseudoStyle()) {
        selectionPaintStyle.hasExplicitlySetFillColor = pseudoStyle->hasExplicitlySetColor();
        selectionShadow = ShadowData::clone(paintInfo.forceTextColor() ? nullptr : pseudoStyle->textShadow());
        auto viewportSize = renderer.frame().view() ? renderer.frame().view()->size() : IntSize();
        float strokeWidth = pseudoStyle->computedStrokeWidth(viewportSize);
        if (strokeWidth != selectionPaintStyle.strokeWidth)
            selectionPaintStyle.strokeWidth = strokeWidth;

        Color stroke = paintInfo.forceTextColor() ? paintInfo.forcedTextColor() : pseudoStyle->computedStrokeColor();
        if (stroke != selectionPaintStyle.strokeColor)
            selectionPaintStyle.strokeColor = stroke;
    } else
        selectionShadow = ShadowData::clone(paintInfo.forceTextColor() ? nullptr : lineStyle.textShadow());
#else
    UNUSED_PARAM(renderer);
    UNUSED_PARAM(lineStyle);
    UNUSED_PARAM(paintInfo);
    selectionShadow = ShadowData::clone(paintInfo.forceTextColor() ? nullptr : lineStyle.textShadow());
#endif
    return selectionPaintStyle;
}

void updateGraphicsContext(GraphicsContext& context, const TextPaintStyle& paintStyle, FillColorType fillColorType)
{
    TextDrawingModeFlags mode = context.textDrawingMode();
    TextDrawingModeFlags newMode = mode;
    if (paintStyle.strokeWidth > 0 && paintStyle.strokeColor.isVisible())
        newMode.add(TextDrawingMode::Stroke);
    if (mode != newMode) {
        context.setTextDrawingMode(newMode);
        mode = newMode;
    }
    context.setUseDarkAppearance(paintStyle.useDarkAppearance);

    Color fillColor = fillColorType == UseEmphasisMarkColor ? paintStyle.emphasisMarkColor : paintStyle.fillColor;
    if (mode.contains(TextDrawingMode::Fill) && (fillColor != context.fillColor()))
        context.setFillColor(fillColor);

    if (mode & TextDrawingMode::Stroke) {
        if (paintStyle.strokeColor != context.strokeColor())
            context.setStrokeColor(paintStyle.strokeColor);
        if (paintStyle.strokeWidth != context.strokeThickness())
            context.setStrokeThickness(paintStyle.strokeWidth);
        context.setLineJoin(paintStyle.lineJoin);
        context.setLineCap(paintStyle.lineCap);
        if (paintStyle.lineJoin == LineJoin::Miter)
            context.setMiterLimit(paintStyle.miterLimit);
    }
}

}
