/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#include "WebCoreTextRenderer.h"

#include "FontCascade.h"
#include "GraphicsContext.h"
#include "TextRun.h"

namespace WebCore {

static bool isOneLeftToRightRun(const TextRun& run)
{
    for (size_t i = 0; i < run.length(); i++) {
        UCharDirection direction = u_charDirection(run[i]);
        if (direction == U_RIGHT_TO_LEFT || direction > U_OTHER_NEUTRAL)
            return false;
    }
    return true;
}

static void doDrawTextAtPoint(GraphicsContext& context, const String& text, const IntPoint& point, const FontCascade& font, const Color& color, int underlinedIndex)
{
    TextRun run(text);

    context.setFillColor(color);
    if (isOneLeftToRightRun(run))
        font.drawText(context, run, point);
    else
        context.drawBidiText(font, run, point);

    if (underlinedIndex >= 0) {
        ASSERT_WITH_SECURITY_IMPLICATION(underlinedIndex < static_cast<int>(text.length()));

        int beforeWidth;
        if (underlinedIndex > 0) {
            TextRun beforeRun(StringView(text).left(underlinedIndex));
            beforeWidth = font.width(beforeRun);
        } else
            beforeWidth = 0;

        TextRun underlinedRun(StringView(text).substring(underlinedIndex, 1));
        int underlinedWidth = font.width(underlinedRun);

        IntPoint underlinePoint(point);
        underlinePoint.move(beforeWidth, 1);

        context.setStrokeColor(color);
        context.drawLineForText(FloatRect(underlinePoint, FloatSize(underlinedWidth, font.size() / 16)), false);
    }
}

void WebCoreDrawDoubledTextAtPoint(GraphicsContext& context, const String& text, const IntPoint& point, const FontCascade& font, const Color& topColor, const Color& bottomColor, int underlinedIndex)
{
    context.save();

    IntPoint textPos = point;

    doDrawTextAtPoint(context, text, textPos, font, bottomColor, underlinedIndex);
    textPos.move(0, -1);
    doDrawTextAtPoint(context, text, textPos, font, topColor, underlinedIndex);

    context.restore();
}

} // namespace WebCore
