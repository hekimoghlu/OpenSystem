/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#include "FloatPoint.h"
#include "InlineTextBoxStyle.h"
#include "RenderStyleConstants.h"
#include <wtf/OptionSet.h>

namespace WebCore {

class FilterOperations;
class FontCascade;
class GraphicsContext;
class RenderObject;
class RenderStyle;
class ShadowData;
class TextRun;
    
class TextDecorationPainter {
public:
    TextDecorationPainter(GraphicsContext&, const FontCascade&, const ShadowData*, const FilterOperations*, bool isPrinting, WritingMode);

    struct Styles {
        bool operator==(const Styles&) const;

        struct DecorationStyleAndColor {
            Color color;
            TextDecorationStyle decorationStyle { TextDecorationStyle::Solid };
        };
        DecorationStyleAndColor underline;
        DecorationStyleAndColor overline;
        DecorationStyleAndColor linethrough;
    };
    struct BackgroundDecorationGeometry {
        FloatPoint textOrigin;
        FloatPoint boxOrigin;
        float textBoxWidth { 0.f };
        float textDecorationThickness { 0.f };
        float underlineOffset { 0.f };
        float overlineOffset { 0.f };
        float linethroughCenter { 0.f };
        float clippingOffset { 0.f };
        WavyStrokeParameters wavyStrokeParameters;
    };
    void paintBackgroundDecorations(const RenderStyle&, const TextRun&, const BackgroundDecorationGeometry&, OptionSet<TextDecorationLine>, const Styles&);

    struct ForegroundDecorationGeometry {
        FloatPoint boxOrigin;
        float textBoxWidth { 0.f };
        float textDecorationThickness { 0.f };
        float linethroughCenter { 0.f };
        WavyStrokeParameters wavyStrokeParameters;
    };
    void paintForegroundDecorations(const ForegroundDecorationGeometry&, const Styles&);

    static Color decorationColor(const RenderStyle&, OptionSet<PaintBehavior> paintBehavior = { });
    static Styles stylesForRenderer(const RenderObject&, OptionSet<TextDecorationLine> requestedDecorations, bool firstLineStyle = false, OptionSet<PaintBehavior> paintBehavior = { }, PseudoId = PseudoId::None);
    static OptionSet<TextDecorationLine> textDecorationsInEffectForStyle(const TextDecorationPainter::Styles&);

private:
    void paintLineThrough(const ForegroundDecorationGeometry&, const Color&, const Styles&);

    GraphicsContext& m_context;
    bool m_isPrinting { false };
    WritingMode m_writingMode;
    const ShadowData* m_shadow { nullptr };
    const FilterOperations* m_shadowColorFilter { nullptr };
    const FontCascade& m_font;
};

} // namespace WebCore
