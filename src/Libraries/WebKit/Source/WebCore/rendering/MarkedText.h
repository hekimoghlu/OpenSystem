/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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

#include <wtf/CheckedPtr.h>
#include <wtf/Hasher.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RenderBoxModelObject;
class RenderText;
class RenderedDocumentMarker;
struct TextBoxSelectableRange;
enum class DocumentMarkerType : uint32_t;

struct MarkedText : public CanMakeCheckedPtr<MarkedText, WTF::DefaultedOperatorEqual::Yes> {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    WTF_STRUCT_OVERRIDE_DELETE_FOR_CHECKED_PTR(MarkedText);

    // Sorted by paint order
    enum class Type : uint8_t {
        Unmarked,
        GrammarError,
        Correction,
#if ENABLE(WRITING_TOOLS)
        WritingToolsTextSuggestion,
#endif
        SpellingError,
        TextMatch,
        DictationAlternatives,
        Highlight,
        FragmentHighlight,
#if ENABLE(APP_HIGHLIGHTS)
        AppHighlight,
#endif
#if PLATFORM(IOS_FAMILY)
        // FIXME: See <rdar://problem/8933352>. Also, remove the PLATFORM(IOS_FAMILY)-guard.
        DictationPhraseWithAlternatives,
#endif
        Selection,
        DraggedContent,
        TransparentContent,
    };

    enum class PaintPhase {
        Background,
        Foreground,
        Decoration
    };

    enum class OverlapStrategy {
        None,
        Frontmost
    };

    MarkedText(unsigned startOffset, unsigned endOffset, Type type, const RenderedDocumentMarker* marker = nullptr, const AtomString& highlightName = { }, int priority = 0) : startOffset(startOffset), endOffset(endOffset), type(type), marker(marker), highlightName(highlightName), priority(priority) { };
    MarkedText(WTF::HashTableDeletedValueType) : startOffset(std::numeric_limits<unsigned>::max()) { };
    MarkedText() = default;

    bool isEmpty() const { return endOffset <= startOffset; }
    bool isHashTableDeletedValue() const { return startOffset == std::numeric_limits<unsigned>::max(); }
    bool operator==(const MarkedText& other) const = default;

    WEBCORE_EXPORT static Vector<MarkedText> subdivide(const Vector<MarkedText>&, OverlapStrategy = OverlapStrategy::None);

    static Vector<MarkedText> collectForDocumentMarkers(const RenderText&, const TextBoxSelectableRange&, PaintPhase);
    static Vector<MarkedText> collectForHighlights(const RenderText&, const TextBoxSelectableRange&, PaintPhase);
    static Vector<MarkedText> collectForDraggedAndTransparentContent(const DocumentMarkerType, const RenderText& renderer, const TextBoxSelectableRange&);

    unsigned startOffset { 0 };
    unsigned endOffset { 0 };
    Type type { Type::Unmarked };
    const RenderedDocumentMarker* marker { nullptr };
    AtomString highlightName;
    int priority { 0 };
};

}
namespace WTF {

inline void add(Hasher& hasher, const WebCore::MarkedText& text)
{
    add(hasher, text.startOffset, text.endOffset, text.type);
}

template<> struct HashTraits<WebCore::MarkedText> : public GenericHashTraits<WebCore::MarkedText> {
    static void constructDeletedValue(WebCore::MarkedText& slot) { slot = WebCore::MarkedText(HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::MarkedText& slot) { return slot.isHashTableDeletedValue(); }
};

template<> struct DefaultHash<WebCore::MarkedText> {
    static unsigned hash(const WebCore::MarkedText& key)
    {
        return computeHash(key);
    }

    static bool equal(const WebCore::MarkedText& a, const WebCore::MarkedText& b)
    {
        return a == b;
    }

    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};
} // namespace WTF

