/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#include "Theme.h"

#include "Color.h"
#include "GraphicsContext.h"
#include "LengthBox.h"
#include "LengthSize.h"

namespace WebCore {

std::optional<FontCascadeDescription> Theme::controlFont(StyleAppearance, const FontCascade&, float) const
{
    return std::nullopt;
}

LengthSize Theme::controlSize(StyleAppearance, const FontCascade&, const LengthSize& zoomedSize, float) const
{
    return zoomedSize;
}

LengthSize Theme::minimumControlSize(StyleAppearance appearance, const FontCascade& fontCascade, const LengthSize& zoomedSize, const LengthSize& nonShrinkableZoomedSize, float zoom) const
{
    auto minSize = minimumControlSize(appearance, fontCascade, zoomedSize, zoom);
    // Other StyleAppearance types are composed controls with shadow subtree.
    if (appearance == StyleAppearance::Radio || appearance == StyleAppearance::Checkbox) {
        if (zoomedSize.width.isIntrinsicOrAuto())
            minSize.width = nonShrinkableZoomedSize.width;
        if (zoomedSize.height.isIntrinsicOrAuto())
            minSize.height = nonShrinkableZoomedSize.height;
    }
    return minSize;
}

LengthSize Theme::minimumControlSize(StyleAppearance, const FontCascade&, const LengthSize&, float) const
{
    return { { 0, LengthType::Fixed }, { 0, LengthType::Fixed } };
}

LengthBox Theme::controlBorder(StyleAppearance appearance, const FontCascade&, const LengthBox& zoomedBox, float) const
{
    switch (appearance) {
    case StyleAppearance::PushButton:
    case StyleAppearance::Menulist:
    case StyleAppearance::SearchField:
    case StyleAppearance::Checkbox:
    case StyleAppearance::Radio:
        return LengthBox(0);
    default:
        return zoomedBox;
    }
}

LengthBox Theme::controlPadding(StyleAppearance appearance, const FontCascade&, const LengthBox& zoomedBox, float) const
{
    switch (appearance) {
    case StyleAppearance::Menulist:
    case StyleAppearance::MenulistButton:
    case StyleAppearance::Checkbox:
    case StyleAppearance::Radio:
        return LengthBox(0);
    default:
        return zoomedBox;
    }
}

void Theme::drawNamedImage(const String& name, GraphicsContext& context, const FloatSize& size) const
{
    // We only handle one icon at the moment.
    if (name != "wireless-playback"_s)
        return;

    GraphicsContextStateSaver stateSaver(context);
    context.setFillColor(Color::black);

    // Draw a generic Wireless Playback icon.

    context.scale(size / 100);
    context.translate(8, 1);

    Path outline;
    outline.moveTo(FloatPoint(59, 58.7));
    outline.addBezierCurveTo(FloatPoint(58.1, 58.7), FloatPoint(57.2, 58.4), FloatPoint(56.4, 57.7));
    outline.addLineTo(FloatPoint(42, 45.5));
    outline.addLineTo(FloatPoint(27.6, 57.8));
    outline.addBezierCurveTo(FloatPoint(25.9, 59.2), FloatPoint(23.4, 59), FloatPoint(22, 57.3));
    outline.addBezierCurveTo(FloatPoint(20.6, 55.6), FloatPoint(20.8, 53.1), FloatPoint(22.5, 51.7));
    outline.addLineTo(FloatPoint(39.5, 37.3));
    outline.addBezierCurveTo(FloatPoint(41, 36), FloatPoint(43.2, 36), FloatPoint(44.7, 37.3));
    outline.addLineTo(FloatPoint(61.7, 51.7));
    outline.addBezierCurveTo(FloatPoint(63.4, 53.1), FloatPoint(63.6, 55.7), FloatPoint(62.2, 57.3));
    outline.addBezierCurveTo(FloatPoint(61.3, 58.2), FloatPoint(60.1, 58.7), FloatPoint(59, 58.7));
    outline.addLineTo(FloatPoint(59, 58.7));
    outline.closeSubpath();

    outline.moveTo(FloatPoint(42, 98));
    outline.addBezierCurveTo(FloatPoint(39.8, 98), FloatPoint(38, 96.3), FloatPoint(38, 94.2));
    outline.addLineTo(FloatPoint(38, 43.6));
    outline.addBezierCurveTo(FloatPoint(38, 41.5), FloatPoint(39.8, 39.8), FloatPoint(42, 39.8));
    outline.addBezierCurveTo(FloatPoint(44.2, 39.8), FloatPoint(46, 41.5), FloatPoint(46, 43.6));
    outline.addLineTo(FloatPoint(46, 94.2));
    outline.addBezierCurveTo(FloatPoint(46, 96.3), FloatPoint(44.2, 98), FloatPoint(42, 98));
    outline.addLineTo(FloatPoint(42, 98));
    outline.closeSubpath();

    outline.moveTo(FloatPoint(83.6, 41.6));
    outline.addBezierCurveTo(FloatPoint(83.6, 18.6), FloatPoint(65, 0), FloatPoint(42, 0));
    outline.addBezierCurveTo(FloatPoint(19, 0), FloatPoint(0.4, 18.6), FloatPoint(0.4, 41.6));
    outline.addBezierCurveTo(FloatPoint(0.4, 62.2), FloatPoint(15, 79.2), FloatPoint(35, 82.6));
    outline.addLineTo(FloatPoint(35, 74.5));
    outline.addBezierCurveTo(FloatPoint(20, 71.2), FloatPoint(8.4, 57.7), FloatPoint(8.4, 41.6));
    outline.addBezierCurveTo(FloatPoint(8.4, 23.1), FloatPoint(23.5, 8), FloatPoint(42, 8));
    outline.addBezierCurveTo(FloatPoint(60.5, 8), FloatPoint(75.5, 23.1), FloatPoint(75.5, 41.6));
    outline.addBezierCurveTo(FloatPoint(75.6, 57.7), FloatPoint(64, 71.2), FloatPoint(49, 74.5));
    outline.addLineTo(FloatPoint(49, 82.6));
    outline.addBezierCurveTo(FloatPoint(69, 79.3), FloatPoint(83.6, 62.2), FloatPoint(83.6, 41.6));
    outline.addLineTo(FloatPoint(83.6, 41.6));
    outline.closeSubpath();

    context.fillPath(outline);
}

}
