/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
#include "ControlStyle.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, ControlStyle::State state)
{
    switch (state) {
    case ControlStyle::State::Hovered:
        ts << "hovered";
        break;
    case ControlStyle::State::Pressed:
        ts << "pressed";
        break;
    case ControlStyle::State::Focused:
        ts << "focused";
        break;
    case ControlStyle::State::Enabled:
        ts << "enabled";
        break;
    case ControlStyle::State::Checked:
        ts << "checked";
        break;
    case ControlStyle::State::Default:
        ts << "default";
        break;
    case ControlStyle::State::WindowActive:
        ts << "window-active";
        break;
    case ControlStyle::State::Indeterminate:
        ts << "indeterminate";
        break;
    case ControlStyle::State::SpinUp:
        ts << "spin-up";
        break;
    case ControlStyle::State::Presenting:
        ts << "presenting";
        break;
    case ControlStyle::State::FormSemanticContext:
        ts << "form-semantic-context";
        break;
    case ControlStyle::State::DarkAppearance:
        ts << "dark-appearance";
        break;
    case ControlStyle::State::InlineFlippedWritingMode:
        ts << "inline-flipped-writing-mode";
        break;
    case ControlStyle::State::LargeControls:
        ts << "large-controls";
        break;
    case ControlStyle::State::ReadOnly:
        ts << "read-only";
        break;
    case ControlStyle::State::ListButton:
        ts << "list-button";
        break;
    case ControlStyle::State::ListButtonPressed:
        ts << "list-button-pressed";
        break;
    case ControlStyle::State::VerticalWritingMode:
        ts << "vertical-writing-mode";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, const ControlStyle& style)
{
    ts.dumpProperty("states", style.states);
    ts.dumpProperty("font-size", style.fontSize);
    ts.dumpProperty("zoom-factor", style.zoomFactor);
    ts.dumpProperty("accent-color", style.accentColor);
    return ts;
}

} // namespace WebCore
