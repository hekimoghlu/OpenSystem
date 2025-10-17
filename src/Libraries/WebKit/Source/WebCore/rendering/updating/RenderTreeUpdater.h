/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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

#include "RenderTreeBuilder.h"
#include "RenderTreePosition.h"
#include "StyleChange.h"
#include "StyleTreeResolver.h"
#include "StyleUpdate.h"
#include <wtf/Vector.h>

namespace WebCore {

class ContainerNode;
class Document;
class Element;
class Node;
class RenderStyle;
class Text;

class RenderTreeUpdater {
public:
    RenderTreeUpdater(Document&, Style::PostResolutionCallbackDisabler&);
    ~RenderTreeUpdater();

    void commit(std::unique_ptr<Style::Update>);

    static void tearDownRenderers(Element&);
    static void tearDownRenderersForShadowRootInsertion(Element&);
    static void tearDownRenderersAfterSlotChange(Element& host);
    static void tearDownRenderer(Text&);

private:
    class GeneratedContent;
    class ViewTransition;

    void updateRenderTree(ContainerNode& root);
    void updateTextRenderer(Text&, const Style::TextUpdate*, const ContainerNode* root = nullptr);
    void createTextRenderer(Text&, const Style::TextUpdate*);
    void updateElementRenderer(Element&, const Style::ElementUpdate&);
    void updateSVGRenderer(Element&);
    void updateRendererStyle(RenderElement&, RenderStyle&&, StyleDifference);
    void updateRenderViewStyle();
    void createRenderer(Element&, RenderStyle&&);
    void updateBeforeDescendants(Element&, const Style::ElementUpdate*);
    void updateAfterDescendants(Element&, const Style::ElementUpdate*);
    bool textRendererIsNeeded(const Text& textNode);
    void storePreviousRenderer(Node&);

    struct Parent {
        Element* element { nullptr };
        const Style::ElementUpdate* update { nullptr };
        std::optional<RenderTreePosition> renderTreePosition;

        bool didCreateOrDestroyChildRenderer { false };
        RenderObject* previousChildRenderer { nullptr };
        bool hasPrecedingInFlowChild { false };

        Parent(ContainerNode& root);
        Parent(Element&, const Style::ElementUpdate*);
    };
    Parent& parent() { return m_parentStack.last(); }
    Parent& renderingParent();
    RenderTreePosition& renderTreePosition();

    GeneratedContent& generatedContent() { return *m_generatedContent; }
    ViewTransition& viewTransition() { return *m_viewTransition; }

    void pushParent(Element&, const Style::ElementUpdate*);
    void popParent();
    void popParentsToDepth(unsigned depth);

    // FIXME: Use OptionSet.
    enum class TeardownType { Full, FullAfterSlotOrShadowRootChange, RendererUpdate, RendererUpdateCancelingAnimations };
    static void tearDownRenderers(Element&, TeardownType);
    static void tearDownRenderers(Element&, TeardownType, RenderTreeBuilder&);
    enum class NeedsRepaintAndLayout : bool { No, Yes };
    static void tearDownTextRenderer(Text&, const ContainerNode* root, RenderTreeBuilder&, NeedsRepaintAndLayout = NeedsRepaintAndLayout::Yes);
    static void tearDownLeftoverChildrenOfComposedTree(Element&, RenderTreeBuilder&);
    static void tearDownLeftoverPaginationRenderersIfNeeded(Element&, RenderTreeBuilder&);

    void updateRebuildRoots();

    RenderView& renderView();

    Ref<Document> m_document;
    std::unique_ptr<Style::Update> m_styleUpdate;

    Vector<Parent> m_parentStack;

    std::unique_ptr<GeneratedContent> m_generatedContent;
    std::unique_ptr<ViewTransition> m_viewTransition;

    RenderTreeBuilder m_builder;
};

} // namespace WebCore
