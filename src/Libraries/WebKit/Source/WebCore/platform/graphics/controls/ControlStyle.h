/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#include "LengthBox.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

struct ControlStyle {
    enum class State {
        Hovered                  = 1 << 0,
        Pressed                  = 1 << 1,
        Focused                  = 1 << 2,
        Enabled                  = 1 << 3,
        Checked                  = 1 << 4,
        Default                  = 1 << 5,
        WindowActive             = 1 << 6,
        Indeterminate            = 1 << 7,
        SpinUp                   = 1 << 8, // Sub-state for HoverState and PressedState.
        Presenting               = 1 << 9,
        FormSemanticContext      = 1 << 10,
        DarkAppearance           = 1 << 11,
        InlineFlippedWritingMode = 1 << 12,
        LargeControls            = 1 << 13,
        ReadOnly                 = 1 << 14,
        ListButton               = 1 << 15,
        ListButtonPressed        = 1 << 16,
        VerticalWritingMode      = 1 << 17,
    };
    OptionSet<State> states;
    float fontSize { 12 };
    float zoomFactor { 1 };
    Color accentColor;
    Color textColor;
    FloatBoxExtent borderWidth;
};

WEBCORE_EXPORT TextStream& operator<<(TextStream&, ControlStyle::State);
WEBCORE_EXPORT TextStream& operator<<(TextStream&, const ControlStyle&);

} // namespace WebCore
