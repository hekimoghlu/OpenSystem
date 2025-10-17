/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#include "RenderStyle.h"
#include <wtf/HashFunctions.h>
#include <wtf/HashMap.h>
#include <wtf/Hasher.h>

namespace WebCore {
namespace Layout {

enum class WhiteSpaceCollapseBehavior : uint8_t {
    CollapseAll,
    CollapseWhitespaceSequence,
    Preserve
};

static WhiteSpaceCollapseBehavior BehaviorForWhiteSpaceCollapse(WhiteSpaceCollapse collapse)
{
    switch (collapse) {
    case WhiteSpaceCollapse::Collapse:
        return WhiteSpaceCollapseBehavior::CollapseAll;
    case WhiteSpaceCollapse::PreserveBreaks:
        return WhiteSpaceCollapseBehavior::CollapseWhitespaceSequence;
    case WhiteSpaceCollapse::Preserve:
    case WhiteSpaceCollapse::BreakSpaces:
        // TODO: WhiteSpaceCollapse::PreserveSpaces should also be handled this way.
        // WhiteSpaceCollapse::Preserve and WhiteSpaceCollapse::BreakSpaces
        // have the same text breaking positions so we don't need to recalculate
        // breaking points when switching between these behaviors.
        return WhiteSpaceCollapseBehavior::Preserve;
    }
    ASSERT_NOT_REACHED();
    return WhiteSpaceCollapseBehavior::CollapseAll;
}

struct TextBreakingPositionContext {
    WhiteSpaceCollapseBehavior whitespaceCollapseBehavior { WhiteSpaceCollapseBehavior::CollapseAll };
    OverflowWrap overflowWrap { OverflowWrap::Normal };
    LineBreak lineBreak { LineBreak::Normal };
    WordBreak wordBreak { WordBreak::Normal };
    NBSPMode nbspMode { NBSPMode::Normal };
    AtomString locale;

    bool isHashTableDeletedValue { false };

    TextBreakingPositionContext(const RenderStyle&);
    TextBreakingPositionContext() = default;

    friend bool operator==(const TextBreakingPositionContext&, const TextBreakingPositionContext&) = default;
};

inline TextBreakingPositionContext::TextBreakingPositionContext(const RenderStyle& style)
    : whitespaceCollapseBehavior(BehaviorForWhiteSpaceCollapse(style.whiteSpaceCollapse()))
    , overflowWrap(style.overflowWrap())
    , lineBreak(style.lineBreak())
    , wordBreak(style.wordBreak())
    , nbspMode(style.nbspMode())
    , locale(style.computedLocale())
{
}

void add(Hasher&, const TextBreakingPositionContext&);

struct TextBreakingPositionContextHash {
    static unsigned hash(const TextBreakingPositionContext& context) { return computeHash(context); }
    static bool equal(const TextBreakingPositionContext& a, const TextBreakingPositionContext& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace Layout
} // namespace WebCore

namespace WTF {

template<>
struct HashTraits<WebCore::Layout::TextBreakingPositionContext> : GenericHashTraits<WebCore::Layout::TextBreakingPositionContext> {
    static void constructDeletedValue(WebCore::Layout::TextBreakingPositionContext& slot) { slot.isHashTableDeletedValue = true; }
    static bool isDeletedValue(const WebCore::Layout::TextBreakingPositionContext& value) { return value.isHashTableDeletedValue; }
    static WebCore::Layout::TextBreakingPositionContext emptyValue() { return { }; }
};

template<> struct DefaultHash<WebCore::Layout::TextBreakingPositionContext> : WebCore::Layout::TextBreakingPositionContextHash { };

} // namespace WTF
