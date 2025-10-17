/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

#include "BlockEllipsis.h"
#include "Length.h"
#include "ListStyleType.h"
#include "RenderStyleConstants.h"
#include "ScrollbarColor.h"
#include "StyleColor.h"
#include "StyleCustomPropertyData.h"
#include "StyleTextEdge.h"
#include "TabSize.h"
#include "TextUnderlineOffset.h"
#include "TouchAction.h"
#include <wtf/DataRef.h>
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/text/AtomString.h>

#if ENABLE(TEXT_AUTOSIZING)
#include "TextSizeAdjustment.h"
#endif

#if ENABLE(DARK_MODE_CSS)
#include "StyleColorScheme.h"
#endif

namespace WTF {
class TextStream;
}

namespace WebCore {

class CursorList;
class QuotesData;
class ShadowData;
class StyleFilterData;
class StyleImage;

// This struct is for rarely used inherited CSS3, CSS2, and WebKit-specific properties.
// By grouping them together, we save space, and only allocate this object when someone
// actually uses one of these properties.
DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleRareInheritedData);
class StyleRareInheritedData : public RefCounted<StyleRareInheritedData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleRareInheritedData);
public:
    static Ref<StyleRareInheritedData> create() { return adoptRef(*new StyleRareInheritedData); }
    Ref<StyleRareInheritedData> copy() const;
    ~StyleRareInheritedData();

    bool operator==(const StyleRareInheritedData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleRareInheritedData&) const;
#endif

    bool hasColorFilters() const;

    float textStrokeWidth;

    RefPtr<StyleImage> listStyleImage;

    Style::Color textStrokeColor;
    Style::Color textFillColor;
    Style::Color textEmphasisColor;
    
    Style::Color visitedLinkTextStrokeColor;
    Style::Color visitedLinkTextFillColor;
    Style::Color visitedLinkTextEmphasisColor;

    Style::Color caretColor;
    Style::Color visitedLinkCaretColor;

    Style::Color accentColor;

    std::unique_ptr<ShadowData> textShadow;
    
    RefPtr<CursorList> cursorData;
    Length indent;
    float usedZoom;

    TextUnderlineOffset textUnderlineOffset;

    TextEdge textBoxEdge;
    TextEdge lineFitEdge;
    
    Length wordSpacing;
    float miterLimit;

    DataRef<StyleCustomPropertyData> customProperties;

    // Paged media properties.
    unsigned short widows;
    unsigned short orphans;
    unsigned hasAutoWidows : 1;
    unsigned hasAutoOrphans : 1;
    
    unsigned textSecurity : 2; // TextSecurity
    unsigned userModify : 2; // UserModify (editing)
    unsigned wordBreak : 3; // WordBreak
    unsigned overflowWrap : 2; // OverflowWrap
    unsigned nbspMode : 1; // NBSPMode
    unsigned lineBreak : 3; // LineBreak
    unsigned userSelect : 2; // UserSelect
    unsigned colorSpace : 1; // ColorSpace
    unsigned speakAs : 4 { 0 }; // OptionSet<SpeakAs>
    unsigned hyphens : 2; // Hyphens
    unsigned textCombine : 1; // TextCombine
    unsigned textEmphasisFill : 1; // TextEmphasisFill
    unsigned textEmphasisMark : 3; // TextEmphasisMark
    unsigned textEmphasisPosition : 4; // TextEmphasisPosition
    unsigned textIndentLine : 1; // TextIndentLine
    unsigned textIndentType : 1; // TextIndentType
    unsigned textUnderlinePosition : 4; // TextUnderlinePosition
    unsigned lineBoxContain: 7; // OptionSet<LineBoxContain>
    // CSS Image Values Level 3
    unsigned imageOrientation : 1; // ImageOrientation
    unsigned imageRendering : 3; // ImageRendering
    unsigned lineSnap : 2; // LineSnap
    unsigned lineAlign : 1; // LineAlign
#if ENABLE(OVERFLOW_SCROLLING_TOUCH)
    unsigned useTouchOverflowScrolling: 1;
#endif
    unsigned textAlignLast : 3; // TextAlignLast
    unsigned textJustify : 2; // TextJustify
    unsigned textDecorationSkipInk : 2; // TextDecorationSkipInk
    unsigned rubyPosition : 2; // RubyPosition
    unsigned rubyAlign : 2; // RubyAlign
    unsigned rubyOverhang : 1; // RubyOverhang
    unsigned textZoom: 1; // TextZoom

#if PLATFORM(IOS_FAMILY)
    unsigned touchCalloutEnabled : 1;
#endif

    unsigned hangingPunctuation : 4; // OptionSet<HangingPunctuation>

    unsigned paintOrder : 3; // PaintOrder
    unsigned capStyle : 2; // LineCap
    unsigned joinStyle : 2; // LineJoin
    unsigned hasSetStrokeWidth : 1;
    unsigned hasSetStrokeColor : 1;

    unsigned mathStyle : 1; // MathStyle

    unsigned hasAutoCaretColor : 1;
    unsigned hasVisitedLinkAutoCaretColor : 1;

    unsigned hasAutoAccentColor : 1;

    unsigned effectiveInert : 1;

    unsigned isInSubtreeWithBlendMode : 1;

    unsigned isInVisibilityAdjustmentSubtree : 1;

    unsigned usedContentVisibility : 2; // ContentVisibility

    OptionSet<TouchAction> usedTouchActions;
    OptionSet<EventListenerRegionType> eventListenerRegionTypes;

    Length strokeWidth;
    Style::Color strokeColor;
    Style::Color visitedLinkStrokeColor;

    AtomString hyphenationString;
    short hyphenationLimitBefore { -1 };
    short hyphenationLimitAfter { -1 };
    short hyphenationLimitLines { -1 };

#if ENABLE(DARK_MODE_CSS)
    Style::ColorScheme colorScheme;
#endif

    AtomString textEmphasisCustomMark;
    RefPtr<QuotesData> quotes;
    DataRef<StyleFilterData> appleColorFilter;

    AtomString lineGrid;
    TabSize tabSize;

#if ENABLE(TEXT_AUTOSIZING)
    TextSizeAdjustment textSizeAdjust;
#endif

#if ENABLE(TOUCH_EVENTS)
    Style::Color tapHighlightColor;
#endif

    ListStyleType listStyleType;

    Markable<ScrollbarColor> scrollbarColor;

    BlockEllipsis blockEllipsis;

private:
    StyleRareInheritedData();
    StyleRareInheritedData(const StyleRareInheritedData&);
};

} // namespace WebCore
