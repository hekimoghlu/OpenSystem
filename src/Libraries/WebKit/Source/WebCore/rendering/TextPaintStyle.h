/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#include "GraphicsTypes.h"
#include "RenderStyleConstants.h"

namespace WebCore {

class GraphicsContext;
class LocalFrame;
class RenderStyle;
class RenderText;
class ShadowData;
struct PaintInfo;

struct TextPaintStyle {
    TextPaintStyle() = default;
    TextPaintStyle(const Color&);

    bool operator==(const TextPaintStyle&) const;

    Color fillColor;
    Color strokeColor;
    Color emphasisMarkColor;
    float strokeWidth { 0 };
    // This is not set for -webkit-text-fill-color.
    bool hasExplicitlySetFillColor { false };
    bool useDarkAppearance { false };
    PaintOrder paintOrder { PaintOrder::Normal };
    LineJoin lineJoin { LineJoin::Miter };
    LineCap lineCap { LineCap::Butt };
    float miterLimit { defaultMiterLimit };
};

bool textColorIsLegibleAgainstBackgroundColor(const Color& textColor, const Color& backgroundColor);
TextPaintStyle computeTextPaintStyle(const RenderText&, const RenderStyle&, const PaintInfo&);
TextPaintStyle computeTextSelectionPaintStyle(const TextPaintStyle&, const RenderText&, const RenderStyle&, const PaintInfo&, std::optional<ShadowData>& selectionShadow);

enum FillColorType { UseNormalFillColor, UseEmphasisMarkColor };
void updateGraphicsContext(GraphicsContext&, const TextPaintStyle&, FillColorType = UseNormalFillColor);

} // namespace WebCore
