/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
#include "config.h"
#include "StyleUpdate.h"

#include "ComposedTreeAncestorIterator.h"
#include "Document.h"
#include "Element.h"
#include "NodeRenderStyle.h"
#include "RenderElement.h"
#include "SVGElement.h"
#include "Text.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Style {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Update);

Update::Update(Document& document)
    : m_document(document)
{
}

Update::~Update() = default;

const ElementUpdate* Update::elementUpdate(const Element& element) const
{
    auto it = m_elements.find(&element);
    if (it == m_elements.end())
        return nullptr;
    return &it->value;
}

ElementUpdate* Update::elementUpdate(const Element& element)
{
    auto it = m_elements.find(&element);
    if (it == m_elements.end())
        return nullptr;
    return &it->value;
}

const TextUpdate* Update::textUpdate(const Text& text) const
{
    auto it = m_texts.find(&text);
    if (it == m_texts.end())
        return nullptr;
    return &it->value;
}

const RenderStyle* Update::elementStyle(const Element& element) const
{
    if (auto* update = elementUpdate(element))
        return update->style.get();
    auto* renderer = element.renderer();
    if (!renderer)
        return nullptr;
    return &renderer->style();
}

RenderStyle* Update::elementStyle(const Element& element)
{
    if (auto* update = elementUpdate(element))
        return update->style.get();
    auto* renderer = element.renderer();
    if (!renderer)
        return nullptr;
    return &renderer->mutableStyle();
}

void Update::addElement(Element& element, Element* parent, ElementUpdate&& elementUpdate)
{
    ASSERT(composedTreeAncestors(element).first() == parent);
    ASSERT(!m_elements.contains(&element));

    m_roots.remove(&element);
    addPossibleRoot(parent);

    if (elementUpdate.mayNeedRebuildRoot)
        addPossibleRebuildRoot(element, parent);

    m_elements.add(&element, WTFMove(elementUpdate));
}

void Update::addText(Text& text, Element* parent, TextUpdate&& textUpdate)
{
    ASSERT(composedTreeAncestors(text).first() == parent);

    addPossibleRoot(parent);

    auto result = m_texts.add(&text, WTFMove(textUpdate));

    if (!result.isNewEntry) {
        auto& entry = result.iterator->value;
        auto startOffset = std::min(entry.offset, textUpdate.offset);
        auto endOffset = std::max(entry.offset + entry.length, textUpdate.offset + textUpdate.length);
        entry.offset = startOffset;
        entry.length = endOffset - startOffset;
        
        ASSERT(!entry.inheritedDisplayContentsStyle || !textUpdate.inheritedDisplayContentsStyle);
        if (!entry.inheritedDisplayContentsStyle)
            entry.inheritedDisplayContentsStyle = WTFMove(textUpdate.inheritedDisplayContentsStyle);
    }
}

void Update::addText(Text& text, TextUpdate&& textUpdate)
{
    addText(text, composedTreeAncestors(text).first(), WTFMove(textUpdate));
}

void Update::addSVGRendererUpdate(SVGElement& element)
{
    auto parent = composedTreeAncestors(element).first();
    m_roots.remove(&element);
    addPossibleRoot(parent);
    element.setNeedsSVGRendererUpdate(true);
}

void Update::addInitialContainingBlockUpdate(std::unique_ptr<RenderStyle> style)
{
    m_initialContainingBlockUpdate = WTFMove(style);
}

void Update::addPossibleRoot(Element* element)
{
    if (!element) {
        m_roots.add(m_document.ptr());
        return;
    }
    if (element->needsSVGRendererUpdate() || m_elements.contains(element))
        return;
    m_roots.add(element);
}

void Update::addPossibleRebuildRoot(Element& element, Element* parent)
{
    if (parent && m_rebuildRoots.contains(parent))
        return;

    m_rebuildRoots.add(&element);
}

}
}
