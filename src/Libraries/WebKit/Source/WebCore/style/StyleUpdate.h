/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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

#include "Node.h"
#include "StyleChange.h"
#include <wtf/HashMap.h>
#include <wtf/ListHashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ContainerNode;
class Document;
class Element;
class Node;
class RenderStyle;
class SVGElement;
class Text;

namespace Style {

struct ElementUpdate {
    std::unique_ptr<RenderStyle> style;
    Change change { Change::None };
    bool recompositeLayer { false };
    bool mayNeedRebuildRoot { false };
};

struct TextUpdate {
    unsigned offset { 0 };
    unsigned length { std::numeric_limits<unsigned>::max() };
    std::optional<std::unique_ptr<RenderStyle>> inheritedDisplayContentsStyle;
};

class Update final : public CanMakeCheckedPtr<Update> {
    WTF_MAKE_TZONE_ALLOCATED(Update);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(Update);
public:
    Update(Document&);
    ~Update();

    const ListHashSet<RefPtr<ContainerNode>>& roots() const { return m_roots; }
    ListHashSet<RefPtr<Element>> takeRebuildRoots() { return WTFMove(m_rebuildRoots); }

    const ElementUpdate* elementUpdate(const Element&) const;
    ElementUpdate* elementUpdate(const Element&);

    const TextUpdate* textUpdate(const Text&) const;

    const RenderStyle* initialContainingBlockUpdate() const { return m_initialContainingBlockUpdate.get(); }

    const RenderStyle* elementStyle(const Element&) const;
    RenderStyle* elementStyle(const Element&);

    const Document& document() const { return m_document; }

    bool isEmpty() const { return !size(); }
    unsigned size() const { return m_elements.size() + m_texts.size(); }

    void addElement(Element&, Element* parent, ElementUpdate&&);
    void addText(Text&, Element* parent, TextUpdate&&);
    void addText(Text&, TextUpdate&&);
    void addSVGRendererUpdate(SVGElement&);
    void addInitialContainingBlockUpdate(std::unique_ptr<RenderStyle>);

private:
    void addPossibleRoot(Element*);
    void addPossibleRebuildRoot(Element&, Element* parent);

    Ref<Document> m_document;
    ListHashSet<RefPtr<ContainerNode>> m_roots;
    ListHashSet<RefPtr<Element>> m_rebuildRoots;
    UncheckedKeyHashMap<RefPtr<const Element>, ElementUpdate> m_elements;
    UncheckedKeyHashMap<RefPtr<const Text>, TextUpdate> m_texts;
    std::unique_ptr<RenderStyle> m_initialContainingBlockUpdate;
};

}
}
