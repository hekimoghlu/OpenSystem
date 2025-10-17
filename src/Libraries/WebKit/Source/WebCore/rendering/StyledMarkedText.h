/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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

#include "MarkedText.h"
#include "ShadowData.h"
#include "TextDecorationPainter.h"
#include "TextPaintStyle.h"

namespace WebCore {

class RenderText;
class RenderedDocumentMarker;

struct StyledMarkedText final : MarkedText {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    WTF_STRUCT_OVERRIDE_DELETE_FOR_CHECKED_PTR(StyledMarkedText);

    struct Style {
        Color backgroundColor;
        TextPaintStyle textStyles;
        TextDecorationPainter::Styles textDecorationStyles;
        std::optional<ShadowData> textShadow;
        float alpha { 1 };
    };

    StyledMarkedText(const MarkedText& marker)
        : MarkedText { marker }
    {
    }

    StyledMarkedText(const MarkedText& marker, Style style)
        : MarkedText { marker }
        , style(style)
    {
    }

    Style style;

    static Vector<StyledMarkedText> subdivideAndResolve(const Vector<MarkedText>&, const RenderText&, bool isFirstLine, const PaintInfo&);
    static Vector<StyledMarkedText> coalesceAdjacentWithEqualBackground(const Vector<StyledMarkedText>&);
    static Vector<StyledMarkedText> coalesceAdjacentWithEqualForeground(const Vector<StyledMarkedText>&);
    static Vector<StyledMarkedText> coalesceAdjacentWithEqualDecorations(const Vector<StyledMarkedText>&);
    static Style computeStyleForUnmarkedMarkedText(const RenderText&, const RenderStyle&, bool isFirstLine, const PaintInfo&);
};

}
