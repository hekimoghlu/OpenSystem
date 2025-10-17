/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#import "config.h"
#import "TextFieldMac.h"

#if PLATFORM(MAC)

#import "ControlFactoryMac.h"
#import "GraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "TextFieldPart.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextFieldMac);

TextFieldMac::TextFieldMac(TextFieldPart& owningPart, ControlFactoryMac& controlFactory, NSTextFieldCell* textFieldCell)
    : ControlMac(owningPart, controlFactory)
    , m_textFieldCell(textFieldCell)
{
    ASSERT(m_textFieldCell);
}

bool TextFieldMac::shouldPaintCustomTextField(const ControlStyle& style)
{
    // <rdar://problem/88948646> Prevent AppKit from painting text fields in the light appearance
    // with increased contrast, as the border is not painted, rendering the control invisible.
    return userPrefersContrast() && !style.states.contains(ControlStyle::State::DarkAppearance);
}

void TextFieldMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    FloatRect paintRect(borderRect.rect());

    const auto& states = style.states;
    auto enabled = states.contains(ControlStyle::State::Enabled) && !states.contains(ControlStyle::State::ReadOnly);

    if (shouldPaintCustomTextField(style)) {
        constexpr int strokeThickness = 1;

        FloatRect strokeRect(paintRect);
        strokeRect.inflate(-strokeThickness / 2.0f);

        context.setStrokeColor(enabled ? Color::black : Color::darkGray);
        context.setStrokeStyle(StrokeStyle::SolidStroke);
        context.strokeRect(strokeRect, strokeThickness);
    } else {
        // <rdar://problem/22896977> We adjust the paint rect here to account for how AppKit draws the text
        // field cell slightly smaller than the rect we pass to drawWithFrame.
        AffineTransform transform = context.getCTM();
        if (transform.xScale() > 1 || transform.yScale() > 1) {
            paintRect.inflateX(1 / transform.xScale());
            paintRect.inflateY(2 / transform.yScale());
            paintRect.move(0, -1 / transform.yScale());
        }
        
        [m_textFieldCell.get() setEnabled:enabled];

        auto styleForDrawing = style;
        styleForDrawing.states.remove(ControlStyle::State::Focused);

        drawCell(context, paintRect, deviceScaleFactor, styleForDrawing, m_textFieldCell.get(), true);
    }

    drawListButton(context, borderRect.rect(), deviceScaleFactor, style);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
