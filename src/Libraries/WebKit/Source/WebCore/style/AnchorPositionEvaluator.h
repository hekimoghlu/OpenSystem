/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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

#include "CSSValueKeywords.h"
#include "EventTarget.h"
#include "LayoutUnit.h"
#include "ScopedName.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class Document;
class Element;
class LayoutRect;
class RenderBlock;
class RenderBoxModelObject;

namespace Style {

class BuilderState;

enum class AnchorPositionResolutionStage : uint8_t {
    Initial,
    FoundAnchors,
    Resolved,
    Positioned,
};

using AnchorElements = HashMap<AtomString, WeakRef<Element, WeakPtrImplWithEventTargetData>>;

struct AnchorPositionedState {
    WTF_MAKE_TZONE_ALLOCATED(AnchorPositionedState);
public:
    AnchorElements anchorElements;
    UncheckedKeyHashSet<AtomString> anchorNames;
    AnchorPositionResolutionStage stage;
};

using AnchorsForAnchorName = HashMap<AtomString, Vector<SingleThreadWeakRef<const RenderBoxModelObject>>>;

// https://drafts.csswg.org/css-anchor-position-1/#typedef-anchor-size
enum class AnchorSizeDimension : uint8_t {
    Width,
    Height,
    Block,
    Inline,
    SelfBlock,
    SelfInline
};

using AnchorPositionedStates = WeakHashMap<Element, std::unique_ptr<AnchorPositionedState>, WeakPtrImplWithEventTargetData>;

// https://drafts.csswg.org/css-anchor-position-1/#position-try-order-property
enum class PositionTryOrder : uint8_t {
    Normal,
    MostWidth,
    MostHeight,
    MostBlockSize,
    MostInlineSize
};

WTF::TextStream& operator<<(WTF::TextStream&, PositionTryOrder);

class AnchorPositionEvaluator {
public:
    // Find the anchor element indicated by `elementName` and update the associated anchor resolution data.
    // Returns nullptr if the anchor element can't be found.
    static RefPtr<Element> findAnchorAndAttemptResolution(const BuilderState&, std::optional<ScopedName> elementName);

    using Side = std::variant<CSSValueID, double>;
    static std::optional<double> evaluate(const BuilderState&, std::optional<ScopedName> elementName, Side);
    static std::optional<double> evaluateSize(const BuilderState&, std::optional<ScopedName> elementName, std::optional<AnchorSizeDimension>);

    static void updateAnchorPositioningStatesAfterInterleavedLayout(const Document&);
    static void cleanupAnchorPositionedState(Element&);
    static void updateSnapshottedScrollOffsets(Document&);

    static LayoutRect computeAnchorRectRelativeToContainingBlock(CheckedRef<const RenderBoxModelObject> anchorBox, const RenderBlock& containingBlock);

private:
    static AnchorElements findAnchorsForAnchorPositionedElement(const Element&, const UncheckedKeyHashSet<AtomString>& anchorNames, const AnchorsForAnchorName&);
};

} // namespace Style

} // namespace WebCore
