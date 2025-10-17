/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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

#include "LayoutElementBox.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderBlockFlow;
class RenderBox;
class RenderElement;
class RenderObject;
class RenderTable;
class RenderView;

namespace Layout {

class InitialContainingBlock;
class LayoutState;

class LayoutTree {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LayoutTree);
public:
    LayoutTree(std::unique_ptr<ElementBox>);
    ~LayoutTree() = default;

    const ElementBox& root() const { return *m_root; }

private:
    std::unique_ptr<ElementBox> m_root;
};

class TreeBuilder {
public:
    static std::unique_ptr<Layout::LayoutTree> buildLayoutTree(const RenderView&);

private:
    TreeBuilder();

    void buildSubTree(const RenderElement& parentRenderer, ElementBox& parentContainer);
    void buildTableStructure(const RenderTable& tableRenderer, ElementBox& tableWrapperBox);
    std::unique_ptr<Box> createLayoutBox(const ElementBox& parentContainer, const RenderObject& childRenderer);

    std::unique_ptr<Box> createReplacedBox(Box::ElementAttributes, ElementBox::ReplacedAttributes&&, RenderStyle&&);
    std::unique_ptr<Box> createTextBox(String text, bool isCombined, bool canUseSimplifiedTextMeasuring, bool canUseSimpleFontCodePath, bool hasPositionDependentContentWidth, bool hasStrongDirectionalityContent, RenderStyle&&);
    std::unique_ptr<ElementBox> createContainer(Box::ElementAttributes, RenderStyle&&);
};

#if ENABLE(TREE_DEBUGGING)
String layoutTreeAsText(const InitialContainingBlock&, const LayoutState*);
void showLayoutTree(const InitialContainingBlock&, const LayoutState*);
void showLayoutTree(const InitialContainingBlock&);
void showInlineTreeAndRuns(TextStream&, const LayoutState&, const ElementBox& inlineFormattingRoot, size_t depth);
void printLayoutTreeForLiveDocuments();
#endif

}
}

