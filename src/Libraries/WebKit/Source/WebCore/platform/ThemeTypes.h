/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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

#include "StyleAppearance.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

enum class SelectionPart : bool {
    Background,
    Foreground
};

enum class ThemeFont : uint8_t {
    CaptionFont,
    IconFont,
    MenuFont,
    MessageBoxFont,
    SmallCaptionFont,
    StatusBarFont,
    MiniControlFont,
    SmallControlFont,
    ControlFont
};

enum class ThemeColor : uint8_t {
    ActiveBorderColor,
    ActiveCaptionColor,
    ActiveTextColor,
    AppWorkspaceColor,
    BackgroundColor,
    ButtonFaceColor,
    ButtonHighlightColor,
    ButtonShadowColor,
    ButtonTextColor,
    CanvasColor,
    CanvasTextColor,
    CaptionTextColor,
    FieldColor,
    FieldTextColor,
    GrayTextColor,
    HighlightColor,
    HighlightTextColor,
    InactiveBorderColor,
    InactiveCaptionColor,
    InactiveCaptionTextColor,
    InfoBackgroundColor,
    InfoTextColor,
    LinkTextColor,
    MatchColor,
    MenuTextColor,
    ScrollbarColor,
    ThreeDDarkShadowColor,
    ThreeDFaceColor,
    ThreeDHighlightColor,
    ThreeDLightShadowColor,
    ThreeDShadowColor,
    VisitedTextColor,
    WindowColor,
    WindowFrameColor,
    WindowTextColor,
    FocusRingColor
};

WTF::TextStream& operator<<(WTF::TextStream&, SelectionPart);
WTF::TextStream& operator<<(WTF::TextStream&, ThemeFont);
WTF::TextStream& operator<<(WTF::TextStream&, ThemeColor);

} // namespace WebCore
