/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#include "WrapContentsInDummySpanCommand.h"

#include "ApplyStyleCommand.h"

namespace WebCore {

WrapContentsInDummySpanCommand::WrapContentsInDummySpanCommand(Element& element)
    : SimpleEditCommand(element.document())
    , m_element(element)
{
}

void WrapContentsInDummySpanCommand::executeApply()
{
    Vector<Ref<Node>> children;
    Ref element = protectedElement();
    for (RefPtr child = element->firstChild(); child; child = child->nextSibling())
        children.append(*child);

    auto dummySpan = protectedDummySpan();
    for (auto& child : children)
        dummySpan->appendChild(child);

    element->appendChild(*dummySpan);
}

void WrapContentsInDummySpanCommand::doApply()
{
    m_dummySpan = createStyleSpanElement(protectedDocument());
    
    executeApply();
}
    
void WrapContentsInDummySpanCommand::doUnapply()
{
    if (!m_dummySpan)
        return;

    Ref element = protectedElement();
    if (!element->hasEditableStyle())
        return;

    Vector<Ref<Node>> children;
    auto dummySpan = protectedDummySpan();
    for (RefPtr child = dummySpan->firstChild(); child; child = child->nextSibling())
        children.append(*child);

    for (auto& child : children)
        element->appendChild(child);

    dummySpan->remove();
}

void WrapContentsInDummySpanCommand::doReapply()
{
    if (!m_dummySpan || !protectedElement()->hasEditableStyle())
        return;

    executeApply();
}

#ifndef NDEBUG
void WrapContentsInDummySpanCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(protectedElement().ptr(), nodes);
    addNodeAndDescendants(protectedDummySpan().get(), nodes);
}
#endif
    
} // namespace WebCore
