/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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

#include "LayerFragment.h"
#include "RenderBlockFlow.h"
#include "RenderFragmentedFlow.h"
#include "VisiblePosition.h"
#include <memory>

namespace WebCore {

class Element;
class RenderBox;
class RenderBoxFragmentInfo;
class RenderFragmentedFlow;

class RenderFragmentContainer : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderFragmentContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderFragmentContainer);
public:
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    void setFragmentedFlowPortionRect(const LayoutRect& rect) { m_fragmentedFlowPortionRect = rect; }
    LayoutRect fragmentedFlowPortionRect() const { return m_fragmentedFlowPortionRect; }
    LayoutRect fragmentedFlowPortionOverflowRect() const;

    LayoutPoint fragmentedFlowPortionLocation() const;

    virtual void attachFragment();
    virtual void detachFragment();

    RenderFragmentedFlow* fragmentedFlow() const { return m_fragmentedFlow.get(); }

    // Valid fragments do not create circular dependencies with other flows.
    bool isValid() const { return m_isValid; }
    void setIsValid(bool valid) { m_isValid = valid; }

    RenderBoxFragmentInfo* renderBoxFragmentInfo(const RenderBox&) const;
    RenderBoxFragmentInfo* setRenderBoxFragmentInfo(const RenderBox&, LayoutUnit logicalLeftInset, LayoutUnit logicalRightInset,
        bool containingBlockChainIsInset);
    std::unique_ptr<RenderBoxFragmentInfo> takeRenderBoxFragmentInfo(const RenderBox*);
    void removeRenderBoxFragmentInfo(const RenderBox&);

    void deleteAllRenderBoxFragmentInfo();

    bool isFirstFragment() const;
    bool isLastFragment() const;
    virtual bool shouldClipFragmentedFlowContent() const;

    // These methods represent the width and height of a "page" and for a RenderFragmentContainer they are just the
    // content width and content height of a fragment. For RenderFragmentContainerSets, however, they will be the width and
    // height of a single column or page in the set.
    virtual LayoutUnit pageLogicalWidth() const;
    virtual LayoutUnit pageLogicalHeight() const;

    LayoutUnit logicalTopOfFragmentedFlowContentRect(const LayoutRect&) const;
    LayoutUnit logicalBottomOfFragmentedFlowContentRect(const LayoutRect&) const;
    LayoutUnit logicalTopForFragmentedFlowContent() const { return logicalTopOfFragmentedFlowContentRect(fragmentedFlowPortionRect()); };
    LayoutUnit logicalBottomForFragmentedFlowContent() const { return logicalBottomOfFragmentedFlowContentRect(fragmentedFlowPortionRect()); };

    // This method represents the logical height of the entire flow thread portion used by the fragment or set.
    // For RenderFragmentContainers it matches logicalPaginationHeight(), but for sets it is the height of all the pages
    // or columns added together.
    virtual LayoutUnit logicalHeightOfAllFragmentedFlowContent() const;

    // The top of the nearest page inside the fragment. For RenderFragmentContainers, this is just the logical top of the
    // flow thread portion we contain. For sets, we have to figure out the top of the nearest column or
    // page.
    virtual LayoutUnit pageLogicalTopForOffset(LayoutUnit offset) const;

    // Whether or not this fragment is a set.
    virtual bool isRenderFragmentContainerSet() const { return false; }
    
    virtual void repaintFragmentedFlowContent(const LayoutRect& repaintRect) const;

    virtual void collectLayerFragments(LayerFragments&, const LayoutRect&, const LayoutRect&) { }

    void addLayoutOverflowForBox(const RenderBox&, const LayoutRect&);
    void addVisualOverflowForBox(const RenderBox&, const LayoutRect&);
    LayoutRect visualOverflowRectForBox(const RenderBox&) const;
    LayoutRect layoutOverflowRectForBoxForPropagation(const RenderBox&);
    LayoutRect visualOverflowRectForBoxForPropagation(const RenderBox&);

    LayoutRect rectFlowPortionForBox(const RenderBox&, const LayoutRect&) const;
    
    void setFragmentObjectsFragmentStyle();
    void restoreFragmentObjectsOriginalStyle();

    bool canHaveChildren() const override { return false; }
    bool canHaveGeneratedChildren() const override { return true; }
    VisiblePosition positionForPoint(const LayoutPoint&, HitTestSource, const RenderFragmentContainer*) override;

    virtual Vector<LayoutRect> fragmentRectsForFlowContentRect(const LayoutRect&) const;

protected:
    RenderFragmentContainer(Type, Element&, RenderStyle&&, RenderFragmentedFlow*);
    RenderFragmentContainer(Type, Document&, RenderStyle&&, RenderFragmentedFlow*);
    virtual ~RenderFragmentContainer();

    void ensureOverflowForBox(const RenderBox&, RefPtr<RenderOverflow>&, bool) const;

    void computePreferredLogicalWidths() override;
    void computeIntrinsicLogicalWidths(LayoutUnit& minLogicalWidth, LayoutUnit& maxLogicalWidth) const override;

    LayoutRect overflowRectForFragmentedFlowPortion(const LayoutRect& fragmentedFlowPortionRect, bool isFirstPortion, bool isLastPortion) const;
    void repaintFragmentedFlowContentRectangle(const LayoutRect& repaintRect, const LayoutRect& fragmentedFlowPortionRect, const LayoutPoint& fragmentLocation, const LayoutRect* fragmentedFlowPortionClipRect = 0) const;

    LayoutRect fragmentedFlowContentRectangle(const LayoutRect&, const LayoutRect& fragmentedFlowPortionRect, const LayoutPoint& fragmentLocation, const LayoutRect* fragmentedFlowPortionClipRect = 0) const;

private:
    ASCIILiteral renderName() const override { return "RenderFragmentContainer"_s; }

    void insertedIntoTree() override;
    void willBeRemovedFromTree() override;

    virtual void installFragmentedFlow();

    LayoutPoint mapFragmentPointIntoFragmentedFlowCoordinates(const LayoutPoint&);

protected:
    SingleThreadWeakPtr<RenderFragmentedFlow> m_fragmentedFlow;

private:
    LayoutRect m_fragmentedFlowPortionRect;

    // This map holds unique information about a block that is split across fragments.
    // A RenderBoxFragmentInfo* tells us about any layout information for a RenderBox that
    // is unique to the fragment. For now it just holds logical width information for RenderBlocks, but eventually
    // it will also hold a custom style for any box (for fragment styling).
    using RenderBoxFragmentInfoMap = UncheckedKeyHashMap<SingleThreadWeakRef<const RenderBox>, std::unique_ptr<RenderBoxFragmentInfo>>;
    RenderBoxFragmentInfoMap m_renderBoxFragmentInfo;

    bool m_isValid { false };
};

class CurrentRenderFragmentContainerMaintainer {
    WTF_MAKE_NONCOPYABLE(CurrentRenderFragmentContainerMaintainer);
public:
    CurrentRenderFragmentContainerMaintainer(RenderFragmentContainer&);
    ~CurrentRenderFragmentContainerMaintainer();

    RenderFragmentContainer& fragment() const { return m_fragment; }
private:
    RenderFragmentContainer& m_fragment;
};

#ifndef NDEBUG
TextStream& operator<<(TextStream&, const RenderFragmentContainer&);
#endif

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderFragmentContainer, isRenderFragmentContainer())
