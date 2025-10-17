/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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

#include "FontSizeAdjust.h"
#include "Settings.h"

namespace WebCore {

class Document;
class RenderStyle;
class FontMetrics;
struct FontSizeAdjust;

namespace Style {

enum class MinimumFontSizeRule : uint8_t { None, Absolute, AbsoluteAndRelative };

float computedFontSizeFromSpecifiedSize(float specifiedSize, bool isAbsoluteSize, float zoomFactor, MinimumFontSizeRule, const Settings::Values&);
float computedFontSizeFromSpecifiedSize(float specifiedSize, bool isAbsoluteSize, bool useSVGZoomRules, const RenderStyle*, const Document&);
float computedFontSizeFromSpecifiedSizeForSVGInlineText(float specifiedSize, bool isAbsoluteSize, float zoomFactor, const Document&);
float adjustedFontSize(float size, const FontSizeAdjust&, const FontMetrics&);

// Given a CSS keyword id in the range (CSSValueXxSmall to CSSValueXxxLarge), this function will return
// the correct font size scaled relative to the user's default (medium).
float fontSizeForKeyword(unsigned keywordID, bool shouldUseFixedDefaultSize, const Settings::Values&, bool inQuirksMode = false);
float fontSizeForKeyword(unsigned keywordID, bool shouldUseFixedDefaultSize, const Document&);

// Given a font size in pixel, this function will return legacy font size between 1 and 7.
int legacyFontSizeForPixelSize(int pixelFontSize, bool shouldUseFixedDefaultSize, const Document&);

}
}
