/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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

#include "Font.h"
#include "InlineItem.h"
#include "InlineLine.h"
#include "LayoutUnits.h"
#include <wtf/Range.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/TextBreakIterator.h>

namespace WebCore {

namespace TextSpacing {
struct SpacingState;
}

class RenderStyle;
class TextRun;

namespace Layout {

struct ExpansionInfo;
class InlineTextBox;
class InlineTextItem;

class TextUtil {
public:
    enum class UseTrailingWhitespaceMeasuringOptimization : bool { No, Yes };
    static InlineLayoutUnit width(const InlineTextItem&, const FontCascade&, InlineLayoutUnit contentLogicalLeft);
    static InlineLayoutUnit width(const InlineTextItem&, const FontCascade&, unsigned from, unsigned to, InlineLayoutUnit contentLogicalLeft, UseTrailingWhitespaceMeasuringOptimization = UseTrailingWhitespaceMeasuringOptimization::Yes, TextSpacing::SpacingState spacingState = { });
    static InlineLayoutUnit width(const InlineTextBox&, const FontCascade&, unsigned from, unsigned to, InlineLayoutUnit contentLogicalLeft, UseTrailingWhitespaceMeasuringOptimization = UseTrailingWhitespaceMeasuringOptimization::Yes, TextSpacing::SpacingState spacingState = { });

    static InlineLayoutUnit trailingWhitespaceWidth(const InlineTextBox&, const FontCascade&, size_t startPosition, size_t endPosition);

    using FallbackFontList = SingleThreadWeakHashSet<const Font>;
    enum class IncludeHyphen : bool { No, Yes };
    static FallbackFontList fallbackFontsForText(StringView, const RenderStyle&, IncludeHyphen);

    struct EnclosingAscentDescent {
        InlineLayoutUnit ascent { 0.f };
        InlineLayoutUnit descent { 0.f };
    };
    enum class ShouldUseSimpleGlyphOverflowCodePath : bool { No, Yes };
    static EnclosingAscentDescent enclosingGlyphBoundsForText(StringView, const RenderStyle&, ShouldUseSimpleGlyphOverflowCodePath);

    struct WordBreakLeft {
        size_t length { 0 };
        InlineLayoutUnit logicalWidth { 0 };
    };
    static WordBreakLeft breakWord(const InlineTextBox&, size_t start, size_t length, InlineLayoutUnit width, InlineLayoutUnit availableWidth, InlineLayoutUnit contentLogicalLeft, const FontCascade&);
    static WordBreakLeft breakWord(const InlineTextItem&, const FontCascade&, InlineLayoutUnit textWidth, InlineLayoutUnit availableWidth, InlineLayoutUnit contentLogicalLeft);

    static bool mayBreakInBetween(const InlineTextItem& previousInlineItem, const InlineTextItem& nextInlineItem);
    // FIXME: Remove when computeInlinePreferredLogicalWidths is all IFC.
    static bool mayBreakInBetween(String previousContent, const RenderStyle& previousContentStyle, String nextContent, const RenderStyle& nextContentStyle);
    static unsigned findNextBreakablePosition(CachedLineBreakIteratorFactory&, unsigned startPosition, const RenderStyle&);
    static TextBreakIterator::LineMode::Behavior lineBreakIteratorMode(LineBreak);
    static TextBreakIterator::ContentAnalysis contentAnalysis(WordBreak);

    static bool shouldPreserveSpacesAndTabs(const Box&);
    static bool shouldPreserveNewline(const Box&);
    static bool isWrappingAllowed(const RenderStyle&);
    static bool shouldTrailingWhitespaceHang(const RenderStyle&);

    static bool isStrongDirectionalityCharacter(char32_t);
    static bool containsStrongDirectionalityText(StringView);

    static AtomString ellipsisTextInInlineDirection(bool isHorizontal = true);

    static InlineLayoutUnit hyphenWidth(const RenderStyle&);

    static size_t firstUserPerceivedCharacterLength(const InlineTextItem&);
    static size_t firstUserPerceivedCharacterLength(const InlineTextBox&, size_t startPosition, size_t length);
    static TextDirection directionForTextContent(StringView);

    static bool hasHangablePunctuationStart(const InlineTextItem&, const RenderStyle&);
    static float hangablePunctuationStartWidth(const InlineTextItem&, const RenderStyle&);

    static bool hasHangablePunctuationEnd(const InlineTextItem&, const RenderStyle&);
    static float hangablePunctuationEndWidth(const InlineTextItem&, const RenderStyle&);

    static bool hasHangableStopOrCommaEnd(const InlineTextItem&, const RenderStyle&);
    static float hangableStopOrCommaEndWidth(const InlineTextItem&, const RenderStyle&);

    static bool canUseSimplifiedTextMeasuring(StringView, const FontCascade&, bool whitespaceIsCollapsed, const RenderStyle* firstLineStyle);
    static bool hasPositionDependentContentWidth(StringView);


    static char32_t baseCharacterFromGraphemeCluster(StringView graphemeCluster);
    static char32_t lastBaseCharacterFromText(StringView);
};

} // namespace Layout
} // namespace WebCore
