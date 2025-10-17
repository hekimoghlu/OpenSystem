/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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

#if USE(THEME_ADWAITA)

#include "Color.h"
#include <wtf/Forward.h>

namespace WebCore {
class GraphicsContext;
class FloatRect;
class Path;

namespace Adwaita {

constexpr unsigned focusLineWidth = 2;
constexpr double focusRingOpacity = 0.8;
constexpr double disabledOpacity = 0.5;

constexpr auto arrowColorLight = SRGBA<uint8_t> { 46, 52, 54 };
constexpr auto arrowColorDark = SRGBA<uint8_t> { 238, 238, 236 };

constexpr int textFieldBorderSize = 1;
constexpr auto textFieldBorderColorLight = SRGBA<uint8_t> { 0, 0, 0, 50 };
constexpr auto textFieldBackgroundColorLight = Color::white;

constexpr auto textFieldBorderColorDark = SRGBA<uint8_t> { 255, 255, 255, 50 };
constexpr auto textFieldBackgroundColorDark = SRGBA<uint8_t> { 45, 45, 45 };

constexpr unsigned arrowSize = 16;
constexpr unsigned menuListButtonArrowSize = 16;
constexpr int menuListButtonFocusOffset = -2;
constexpr unsigned menuListButtonPadding = 5;
constexpr int menuListButtonBorderSize = 1;
constexpr unsigned progressActivityBlocks = 5;
constexpr Seconds progressAnimationDuration = 2475_ms;
constexpr unsigned progressBarSize = 6;
constexpr auto progressBarBackgroundColorLight = SRGBA<uint8_t> { 0, 0, 0, 40 };
constexpr auto progressBarBackgroundColorDark = SRGBA<uint8_t> { 255, 255, 255, 30 };
constexpr unsigned sliderTrackSize = 6;
constexpr auto sliderTrackBackgroundColorLight = SRGBA<uint8_t> { 0, 0, 0, 40 };
constexpr auto sliderTrackBackgroundColorDark = SRGBA<uint8_t> { 255, 255, 255, 30 };
constexpr int sliderTrackFocusOffset = 2;
constexpr int sliderThumbSize = 20;
constexpr int sliderThumbBorderSize = 1;
constexpr auto sliderThumbBorderColorLight = SRGBA<uint8_t>  { 0, 0, 0, 50 };
constexpr auto sliderThumbBackgroundColorLight = Color::white;
constexpr auto sliderThumbBackgroundHoveredColorLight = SRGBA<uint8_t> { 244, 244, 244 };
constexpr auto sliderThumbBackgroundDisabledColorLight = SRGBA<uint8_t> { 244, 244, 244 };

constexpr auto sliderThumbBorderColorDark = SRGBA<uint8_t>  { 0, 0, 0, 50 };
constexpr auto sliderThumbBackgroundColorDark = SRGBA<uint8_t> { 210, 210, 210 };
constexpr auto sliderThumbBackgroundHoveredColorDark = SRGBA<uint8_t> { 230, 230, 230 };
constexpr auto sliderThumbBackgroundDisabledColorDark = SRGBA<uint8_t> { 150, 150, 150 };

constexpr int buttonFocusOffset = -2;
constexpr int buttonBorderSize = 1;

constexpr auto buttonTextColorLight = SRGBA<uint8_t> { 0, 0, 0, 204 };
constexpr auto buttonTextDisabledColorLight = SRGBA<uint8_t> { 0, 0, 0, 102 };
constexpr auto buttonTextColorDark = SRGBA<uint8_t> { 255, 255, 255 };
constexpr auto buttonTextDisabledColorDark = SRGBA<uint8_t> { 255, 255, 255, 127 };

constexpr auto buttonBorderColorLight = SRGBA<uint8_t> { 0, 0, 0, 50 };
constexpr auto buttonBackgroundColorLight = SRGBA<uint8_t> { 244, 244, 244 };
constexpr auto buttonBackgroundPressedColorLight = SRGBA<uint8_t> { 214, 214, 214 };
constexpr auto buttonBackgroundHoveredColorLight = SRGBA<uint8_t> { 248, 248, 248 };
constexpr auto toggleBorderColorLight = SRGBA<uint8_t> { 0, 0, 0, 50 };
constexpr auto toggleBorderHoveredColorLight = SRGBA<uint8_t> { 0, 0, 0, 80 };

constexpr auto buttonBorderColorDark = SRGBA<uint8_t> { 255, 255, 255, 50 };
constexpr auto buttonBackgroundColorDark = SRGBA<uint8_t> { 52, 52, 52 };
constexpr auto buttonBackgroundPressedColorDark = SRGBA<uint8_t> { 30, 30, 30 };
constexpr auto buttonBackgroundHoveredColorDark = SRGBA<uint8_t> { 60, 60, 60 };
constexpr auto toggleBorderColorDark = SRGBA<uint8_t> { 255, 255, 255, 50 };
constexpr auto toggleBorderHoveredColorDark = SRGBA<uint8_t> { 255, 255, 255, 80 };

constexpr double toggleSize = 14.;
constexpr int toggleBorderSize = 2;
constexpr int toggleFocusOffset = 1;

constexpr auto spinButtonBorderColorLight = SRGBA<uint8_t> { 0, 0, 0, 25 };
constexpr auto spinButtonBackgroundColorLight = Color::white;
constexpr auto spinButtonBackgroundHoveredColorLight = SRGBA<uint8_t> { 0, 0, 0, 50 };
constexpr auto spinButtonBackgroundPressedColorLight = SRGBA<uint8_t> { 0, 0, 0, 70 };

constexpr auto spinButtonBorderColorDark = SRGBA<uint8_t> { 255, 255, 255, 25 };
constexpr auto spinButtonBackgroundColorDark = SRGBA<uint8_t> { 45, 45, 45 };
constexpr auto spinButtonBackgroundHoveredColorDark = SRGBA<uint8_t> { 255, 255, 255, 50 };
constexpr auto spinButtonBackgroundPressedColorDark = SRGBA<uint8_t> { 255, 255, 255, 70 };


enum class PaintRounded : bool { No, Yes };
void paintFocus(GraphicsContext&, const FloatRect&, int offset, const Color&, PaintRounded = PaintRounded::No);
void paintFocus(GraphicsContext&, const Path&, const Color&);
void paintFocus(GraphicsContext&, const Vector<FloatRect>&, const Color&, PaintRounded = PaintRounded::No);

enum class ArrowDirection { Up, Down };
void paintArrow(GraphicsContext&, const FloatRect&, ArrowDirection, bool useDarkAppearance);

inline Color focusColor(const Color& accentColor)
{
    return accentColor.colorWithAlphaMultipliedBy(focusRingOpacity);
}

Color systemAccentColor();
Color systemFocusRingColor();

} // namespace Adwaita
} // namespace WebCore

#endif // USE(THEME_ADWAITA)
