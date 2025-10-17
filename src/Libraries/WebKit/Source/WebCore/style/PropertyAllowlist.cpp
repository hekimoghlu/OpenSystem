/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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
#include "PropertyAllowlist.h"

namespace WebCore {
namespace Style {

PropertyAllowlist propertyAllowlistForPseudoId(PseudoId pseudoId)
{
    if (pseudoId == PseudoId::Marker)
        return PropertyAllowlist::Marker;
    return PropertyAllowlist::None;
}

// https://drafts.csswg.org/css-lists-3/#marker-properties (Editor's Draft, 14 July 2021)
// FIXME: this is outdated, see https://bugs.webkit.org/show_bug.cgi?id=218791.
bool isValidMarkerStyleProperty(CSSPropertyID id)
{
    switch (id) {
    case CSSPropertyColor:
    case CSSPropertyContent:
    case CSSPropertyCustom:
    case CSSPropertyCursor:
    case CSSPropertyDirection:
    case CSSPropertyFont:
    case CSSPropertyFontFamily:
    case CSSPropertyFontFeatureSettings:
    case CSSPropertyFontKerning:
    case CSSPropertyFontSize:
    case CSSPropertyFontSizeAdjust:
    case CSSPropertyFontWidth:
    case CSSPropertyFontStyle:
    case CSSPropertyFontSynthesis:
    case CSSPropertyFontSynthesisWeight:
    case CSSPropertyFontSynthesisStyle:
    case CSSPropertyFontSynthesisSmallCaps:
    case CSSPropertyFontVariantAlternates:
    case CSSPropertyFontVariantCaps:
    case CSSPropertyFontVariantEastAsian:
    case CSSPropertyFontVariantLigatures:
    case CSSPropertyFontVariantNumeric:
    case CSSPropertyFontVariantPosition:
    case CSSPropertyFontWeight:
#if ENABLE(VARIATION_FONTS)
    case CSSPropertyFontOpticalSizing:
    case CSSPropertyFontVariationSettings:
#endif
    case CSSPropertyHyphens:
    case CSSPropertyLetterSpacing:
    case CSSPropertyLineBreak:
    case CSSPropertyLineHeight:
    case CSSPropertyListStyle:
    case CSSPropertyOverflowWrap:
    case CSSPropertyTabSize:
    case CSSPropertyTextCombineUpright:
    case CSSPropertyTextDecorationSkipInk:
    case CSSPropertyTextEmphasis:
    case CSSPropertyTextEmphasisColor:
    case CSSPropertyTextEmphasisPosition:
    case CSSPropertyTextEmphasisStyle:
    case CSSPropertyTextShadow:
    case CSSPropertyTextTransform:
    case CSSPropertyTextWrapMode:
    case CSSPropertyTextWrapStyle:
    case CSSPropertyUnicodeBidi:
    case CSSPropertyWordBreak:
    case CSSPropertyWordSpacing:
    case CSSPropertyWhiteSpace:
    case CSSPropertyWhiteSpaceCollapse:
    case CSSPropertyAnimationDuration:
    case CSSPropertyAnimationTimingFunction:
    case CSSPropertyAnimationDelay:
    case CSSPropertyAnimationIterationCount:
    case CSSPropertyAnimationDirection:
    case CSSPropertyAnimationFillMode:
    case CSSPropertyAnimationPlayState:
    case CSSPropertyAnimationComposition:
    case CSSPropertyAnimationName:
    case CSSPropertyTransitionBehavior:
    case CSSPropertyTransitionDuration:
    case CSSPropertyTransitionTimingFunction:
    case CSSPropertyTransitionDelay:
    case CSSPropertyTransitionProperty:
        return true;
    default:
        break;
    }
    return false;
}

#if ENABLE(VIDEO)
bool isValidCueStyleProperty(CSSPropertyID id)
{
    switch (id) {
    case CSSPropertyColor:
    case CSSPropertyCustom:
    case CSSPropertyFont:
    case CSSPropertyFontFamily:
    case CSSPropertyFontSize:
    case CSSPropertyFontStyle:
    case CSSPropertyFontVariantCaps:
    case CSSPropertyFontWeight:
    case CSSPropertyLineHeight:
    case CSSPropertyOpacity:
    case CSSPropertyOutline:
    case CSSPropertyOutlineColor:
    case CSSPropertyOutlineOffset:
    case CSSPropertyOutlineStyle:
    case CSSPropertyOutlineWidth:
    case CSSPropertyVisibility:
    case CSSPropertyWhiteSpace:
    case CSSPropertyWhiteSpaceCollapse:
    case CSSPropertyTextCombineUpright:
    case CSSPropertyTextDecorationLine:
    case CSSPropertyTextShadow:
    case CSSPropertyTextWrapMode:
    case CSSPropertyTextWrapStyle:
    case CSSPropertyBorderStyle:
    case CSSPropertyPaintOrder:
    case CSSPropertyStrokeLinejoin:
    case CSSPropertyStrokeLinecap:
    case CSSPropertyStrokeColor:
    case CSSPropertyStrokeWidth:
        return true;
    default:
        break;
    }
    return false;
}
#endif

#if ENABLE(VIDEO)
bool isValidCueSelectorStyleProperty(CSSPropertyID id)
{
    switch (id) {
    case CSSPropertyBackground:
    case CSSPropertyBackgroundAttachment:
    case CSSPropertyBackgroundClip:
    case CSSPropertyBackgroundColor:
    case CSSPropertyBackgroundImage:
    case CSSPropertyBackgroundOrigin:
    case CSSPropertyBackgroundPosition:
    case CSSPropertyBackgroundPositionX:
    case CSSPropertyBackgroundPositionY:
    case CSSPropertyBackgroundRepeat:
    case CSSPropertyBackgroundSize:
    case CSSPropertyColor:
    case CSSPropertyCustom:
    case CSSPropertyFont:
    case CSSPropertyFontFamily:
    case CSSPropertyFontSize:
    case CSSPropertyFontStyle:
    case CSSPropertyFontVariantCaps:
    case CSSPropertyFontWeight:
    case CSSPropertyLineHeight:
    case CSSPropertyOpacity:
    case CSSPropertyOutline:
    case CSSPropertyOutlineColor:
    case CSSPropertyOutlineOffset:
    case CSSPropertyOutlineStyle:
    case CSSPropertyOutlineWidth:
    case CSSPropertyVisibility:
    case CSSPropertyWhiteSpace:
    case CSSPropertyWhiteSpaceCollapse:
    case CSSPropertyTextCombineUpright:
    case CSSPropertyTextDecorationLine:
    case CSSPropertyTextShadow:
    case CSSPropertyTextWrapMode:
    case CSSPropertyTextWrapStyle:
    case CSSPropertyBorderStyle:
    case CSSPropertyPaintOrder:
    case CSSPropertyStrokeLinejoin:
    case CSSPropertyStrokeLinecap:
    case CSSPropertyStrokeColor:
    case CSSPropertyStrokeWidth:
        return true;
    default:
        break;
    }
    return false;
}

bool isValidCueBackgroundStyleProperty(CSSPropertyID id)
{
    switch (id) {
    case CSSPropertyBackground:
    case CSSPropertyBackgroundAttachment:
    case CSSPropertyBackgroundClip:
    case CSSPropertyBackgroundColor:
    case CSSPropertyBackgroundImage:
    case CSSPropertyBackgroundOrigin:
    case CSSPropertyBackgroundPosition:
    case CSSPropertyBackgroundPositionX:
    case CSSPropertyBackgroundPositionY:
    case CSSPropertyBackgroundRepeat:
    case CSSPropertyBackgroundSize:
        return true;
    default:
        break;
    }
    return false;
}
#endif

}
}
