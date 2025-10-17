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

#include <optional>

namespace WebCore {

class CSSToLengthConversionData;
class CSSValue;
class FontCascade;
class FontCascadeDescription;
class FontSelectionValue;
class ScriptExecutionContext;

struct FontSizeAdjust;

template<typename> class FontTaggedSettings;
using FontFeatureSettings = FontTaggedSettings<int>;
using FontVariationSettings = FontTaggedSettings<float>;

namespace CSSPropertyParserHelpers {
struct UnresolvedFont;
}

namespace Style {

class BuilderState;

FontSelectionValue fontWeightFromCSSValueDeprecated(const CSSValue&);
FontSelectionValue fontWeightFromCSSValue(BuilderState&, const CSSValue&);

FontSelectionValue fontStretchFromCSSValueDeprecated(const CSSValue&);
FontSelectionValue fontStretchFromCSSValue(BuilderState&, const CSSValue&);

FontSelectionValue fontStyleAngleFromCSSValueDeprecated(const CSSValue&);
FontSelectionValue fontStyleAngleFromCSSValue(BuilderState&, const CSSValue&);

std::optional<FontSelectionValue> fontStyleFromCSSValueDeprecated(const CSSValue&);
std::optional<FontSelectionValue> fontStyleFromCSSValue(BuilderState&, const CSSValue&);

FontFeatureSettings fontFeatureSettingsFromCSSValue(BuilderState&, const CSSValue&);
FontVariationSettings fontVariationSettingsFromCSSValue(BuilderState&, const CSSValue&);
FontSizeAdjust fontSizeAdjustFromCSSValue(BuilderState&, const CSSValue&);

std::optional<FontCascade> resolveForUnresolvedFont(const CSSPropertyParserHelpers::UnresolvedFont&, FontCascadeDescription&&, ScriptExecutionContext&);

} // namespace Style
} // namespace WebCore
