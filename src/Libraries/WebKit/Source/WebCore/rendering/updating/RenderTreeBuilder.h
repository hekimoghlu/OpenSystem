/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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

#include "RenderTreePosition.h"
#include "RenderWidget.h"

namespace WebCore {

class RenderGrid;
class RenderTreeUpdater;

class RenderTreeBuilder {
public:
    RenderTreeBuilder(RenderView&);
    ~RenderTreeBuilder();

    // This avoids having to convert all sites that need RenderTreeBuilder in one go.
    // FIXME: Remove.
    static RenderTreeBuilder* current() { return s_current; }

    static bool isRebuildRootForChildren(const RenderElement&);

    void attach(RenderElement& parent, RenderPtr<RenderObject>, RenderObject* beforeChild = nullptr);

    enum class IsInternalMove : bool { No, Yes };
    enum class WillBeDestroyed : bool { No, Yes };
    enum class CanCollapseAnonymousBlock : bool { No, Yes };
    RenderPtr<RenderObject> detach(RenderElement&, RenderObject&, WillBeDestroyed, CanCollapseAnonymousBlock = CanCollapseAnonymousBlock::Yes) WARN_UNUSED_RETURN;

    enum class TearDownType : uint8_t {
        Root,                          // destroy root renderer
        SubtreeWithRootStillAttached,  // subtree teardown when renderers are still attached to the tree (common case)
        SubtreeWithRootAlreadyDetached // subtree teardown when destroy root gets detached first followed by destroying renderers (e.g. pseudo subtree)
    };
    void destroy(RenderObject& renderer, CanCollapseAnonymousBlock = CanCollapseAnonymousBlock::Yes);

    // NormalizeAfterInsertion::Yes ensures that the destination subtree is consistent after the insertion (anonymous wrappers etc).
    enum class NormalizeAfterInsertion : bool { No, Yes };
    void move(RenderBoxModelObject& from, RenderBoxModelObject& to, RenderObject& child, NormalizeAfterInsertion);

    void updateAfterDescendants(RenderElement&);
    void destroyAndCleanUpAnonymousWrappers(RenderObject& child, const RenderElement* destroyRoot);
    void normalizeTreeAfterStyleChange(RenderElement&, RenderStyle& oldStyle);

    bool hasBrokenContinuation() const { return m_hasBrokenContinuation; }

private:
    static void markBoxForRelayoutAfterSplit(RenderBox&);

    void attachInternal(RenderElement& parent, RenderPtr<RenderObject>, RenderObject* beforeChild);

    void childFlowStateChangesAndAffectsParentBlock(RenderElement& child);
    void childFlowStateChangesAndNoLongerAffectsParentBlock(RenderElement& child);
    void attachIgnoringContinuation(RenderElement& parent, RenderPtr<RenderObject>, RenderObject* beforeChild = nullptr);
    void attachToRenderGrid(RenderGrid& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild = nullptr);
    void attachToRenderElement(RenderElement& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild = nullptr);
    void attachToRenderElementInternal(RenderElement& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild = nullptr);

    RenderPtr<RenderObject> detachFromRenderElement(RenderElement& parent, RenderObject& child, WillBeDestroyed) WARN_UNUSED_RETURN;
    RenderPtr<RenderObject> detachFromRenderGrid(RenderGrid& parent, RenderObject& child, WillBeDestroyed) WARN_UNUSED_RETURN;

    void move(RenderBoxModelObject& from, RenderBoxModelObject& to, RenderObject& child, RenderObject* beforeChild, NormalizeAfterInsertion);
    // Move all of the kids from |startChild| up to but excluding |endChild|. 0 can be passed as the |endChild| to denote
    // that all the kids from |startChild| onwards should be moved.
    void moveChildren(RenderBoxModelObject& from, RenderBoxModelObject& to, RenderObject* startChild, RenderObject* endChild, NormalizeAfterInsertion);
    void moveChildren(RenderBoxModelObject& from, RenderBoxModelObject& to, RenderObject* startChild, RenderObject* endChild, RenderObject* beforeChild, NormalizeAfterInsertion);
    void moveAllChildrenIncludingFloats(RenderBlock& from, RenderBlock& toBlock, RenderTreeBuilder::NormalizeAfterInsertion);
    void moveAllChildren(RenderBoxModelObject& from, RenderBoxModelObject& to, NormalizeAfterInsertion);
    void moveAllChildren(RenderBoxModelObject& from, RenderBoxModelObject& to, RenderObject* beforeChild, NormalizeAfterInsertion);

    void removeFloatingObjects(RenderBlock&);

    RenderObject* splitAnonymousBoxesAroundChild(RenderBox& parent, RenderObject& originalBeforeChild);
    void createAnonymousWrappersForInlineContent(RenderBlock& parent, RenderObject* insertionPoint = nullptr);
    void removeAnonymousWrappersForInlineChildrenIfNeeded(RenderElement& parent);

    void reportVisuallyNonEmptyContent(const RenderElement& parent, const RenderObject& child);

    void setHasBrokenContinuation() { m_hasBrokenContinuation = true; }

    class FirstLetter;
    class List;
    class MultiColumn;
    class Table;
    class Ruby;
    class FormControls;
    class Block;
    class BlockFlow;
    class Inline;
    class SVG;
#if ENABLE(MATHML)
    class MathML;
#endif
    class Continuation;

    FirstLetter& firstLetterBuilder() { return *m_firstLetterBuilder; }
    List& listBuilder() { return *m_listBuilder; }
    MultiColumn& multiColumnBuilder() { return *m_multiColumnBuilder; }
    Table& tableBuilder() { return *m_tableBuilder; }
    Ruby& rubyBuilder() { return *m_rubyBuilder; }
    FormControls& formControlsBuilder() { return *m_formControlsBuilder; }
    Block& blockBuilder() { return *m_blockBuilder; }
    BlockFlow& blockFlowBuilder() { return *m_blockFlowBuilder; }
    Inline& inlineBuilder() { return *m_inlineBuilder; }
    SVG& svgBuilder() { return *m_svgBuilder; }
#if ENABLE(MATHML)
    MathML& mathMLBuilder() { return *m_mathMLBuilder; }
#endif
    Continuation& continuationBuilder() { return *m_continuationBuilder; }

    WidgetHierarchyUpdatesSuspensionScope m_widgetHierarchyUpdatesSuspensionScope;
    RenderView& m_view;
    RenderTreeBuilder* m_previous { nullptr };
    static RenderTreeBuilder* s_current;

    std::unique_ptr<FirstLetter> m_firstLetterBuilder;
    std::unique_ptr<List> m_listBuilder;
    std::unique_ptr<MultiColumn> m_multiColumnBuilder;
    std::unique_ptr<Table> m_tableBuilder;
    std::unique_ptr<Ruby> m_rubyBuilder;
    std::unique_ptr<FormControls> m_formControlsBuilder;
    std::unique_ptr<Block> m_blockBuilder;
    std::unique_ptr<BlockFlow> m_blockFlowBuilder;
    std::unique_ptr<Inline> m_inlineBuilder;
    std::unique_ptr<SVG> m_svgBuilder;
#if ENABLE(MATHML)
    std::unique_ptr<MathML> m_mathMLBuilder;
#endif
    std::unique_ptr<Continuation> m_continuationBuilder;
    bool m_hasBrokenContinuation { false };
    IsInternalMove m_internalMovesType { IsInternalMove::No };
    TearDownType m_tearDownType { TearDownType::Root };
    CheckedPtr<const RenderElement> m_subtreeDestroyRoot;
    SingleThreadWeakPtr<const RenderObject> m_anonymousDestroyRoot;
};

}
