/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#include "ThemeTypes.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, SelectionPart selectionPart)
{
    switch (selectionPart) {
    case SelectionPart::Background: ts << "selection-background"; break;
    case SelectionPart::Foreground: ts << "selection-foreground"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ThemeFont themeFont)
{
    switch (themeFont) {
    case ThemeFont::CaptionFont: ts << "caption-font"; break;
    case ThemeFont::IconFont: ts << "icon-font"; break;
    case ThemeFont::MenuFont: ts << "menu-font"; break;
    case ThemeFont::MessageBoxFont: ts << "messagebox-font"; break;
    case ThemeFont::SmallCaptionFont: ts << "small-caption-font"; break;
    case ThemeFont::StatusBarFont: ts << "statusbar-font"; break;
    case ThemeFont::MiniControlFont: ts << "minicontrol-font"; break;
    case ThemeFont::SmallControlFont: ts << "small-control-font"; break;
    case ThemeFont::ControlFont: ts << "control-font"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ThemeColor themeColor)
{
    switch (themeColor) {
    case ThemeColor::ActiveBorderColor: ts << "active-border-color"; break;
    case ThemeColor::ActiveCaptionColor: ts << "active-caption-color"; break;
    case ThemeColor::ActiveTextColor: ts << "active-text-color"; break;
    case ThemeColor::AppWorkspaceColor: ts << "app-workspace-color"; break;
    case ThemeColor::BackgroundColor: ts << "background-color"; break;
    case ThemeColor::ButtonFaceColor: ts << "button-face-color"; break;
    case ThemeColor::ButtonHighlightColor: ts << "button-highlight-color"; break;
    case ThemeColor::ButtonShadowColor: ts << "button-shadow-color"; break;
    case ThemeColor::ButtonTextColor: ts << "button-text-color"; break;
    case ThemeColor::CanvasColor: ts << "canvas-color"; break;
    case ThemeColor::CanvasTextColor: ts << "canvas-text-color"; break;
    case ThemeColor::CaptionTextColor: ts << "caption-text-color"; break;
    case ThemeColor::FieldColor: ts << "field-color"; break;
    case ThemeColor::FieldTextColor: ts << "field-text-color"; break;
    case ThemeColor::GrayTextColor: ts << "gray-text-color"; break;
    case ThemeColor::HighlightColor: ts << "highlight-color"; break;
    case ThemeColor::HighlightTextColor: ts << "highlight-text-color"; break;
    case ThemeColor::InactiveBorderColor: ts << "inactive-border-color"; break;
    case ThemeColor::InactiveCaptionColor: ts << "inactive-caption-color"; break;
    case ThemeColor::InactiveCaptionTextColor: ts << "inactive-caption-text-color"; break;
    case ThemeColor::InfoBackgroundColor: ts << "info-background-color"; break;
    case ThemeColor::InfoTextColor: ts << "info-text-color"; break;
    case ThemeColor::LinkTextColor: ts << "link-text-color"; break;
    case ThemeColor::MatchColor: ts << "match-color"; break;
    case ThemeColor::MenuTextColor: ts << "menu-text-color"; break;
    case ThemeColor::ScrollbarColor: ts << "scrollbar-color"; break;
    case ThemeColor::ThreeDDarkShadowColor: ts << "threeD-dark-shadow-color"; break;
    case ThemeColor::ThreeDFaceColor: ts << "threeD-face-color"; break;
    case ThemeColor::ThreeDHighlightColor: ts << "threeD-highlight-color"; break;
    case ThemeColor::ThreeDLightShadowColor: ts << "threeD-light-shadow-color"; break;
    case ThemeColor::ThreeDShadowColor: ts << "threeD-shadow-color"; break;
    case ThemeColor::VisitedTextColor: ts << "visited-text-color"; break;
    case ThemeColor::WindowColor: ts << "window-color"; break;
    case ThemeColor::WindowFrameColor: ts << "window-frame-color"; break;
    case ThemeColor::WindowTextColor: ts << "window-text-color"; break;
    case ThemeColor::FocusRingColor: ts << "focus-ring-color"; break;
    }
    return ts;
}

} // namespace WebCore
