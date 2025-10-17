/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#include "RenderTreeBuilderFormControls.h"

#include "RenderButton.h"
#include "RenderMenuList.h"
#include "RenderTreeBuilderBlock.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderTreeBuilder::FormControls);

RenderTreeBuilder::FormControls::FormControls(RenderTreeBuilder& builder)
    : m_builder(builder)
{
}

void RenderTreeBuilder::FormControls::attach(RenderButton& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild)
{
    m_builder.blockBuilder().attach(findOrCreateParentForChild(parent), WTFMove(child), beforeChild);
}

void RenderTreeBuilder::FormControls::attach(RenderMenuList& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild)
{
    auto& newChild = *child.get();
    m_builder.blockBuilder().attach(findOrCreateParentForChild(parent), WTFMove(child), beforeChild);
    parent.didAttachChild(newChild, beforeChild);
}

RenderPtr<RenderObject> RenderTreeBuilder::FormControls::detach(RenderMenuList& parent, RenderObject& child, RenderTreeBuilder::WillBeDestroyed willBeDestroyed)
{
    auto* innerRenderer = parent.innerRenderer();
    if (!innerRenderer || &child == innerRenderer)
        return m_builder.blockBuilder().detach(parent, child, willBeDestroyed);
    return m_builder.detach(*innerRenderer, child, willBeDestroyed);
}

RenderPtr<RenderObject> RenderTreeBuilder::FormControls::detach(RenderButton& parent, RenderObject& child, RenderTreeBuilder::WillBeDestroyed willBeDestroyed)
{
    auto* innerRenderer = parent.innerRenderer();
    if (!innerRenderer || &child == innerRenderer || child.parent() == &parent) {
        ASSERT(&child == innerRenderer || !innerRenderer);
        return m_builder.blockBuilder().detach(parent, child, willBeDestroyed);
    }
    return m_builder.detach(*innerRenderer, child, willBeDestroyed);
}


RenderBlock& RenderTreeBuilder::FormControls::findOrCreateParentForChild(RenderButton& parent)
{
    auto* innerRenderer = parent.innerRenderer();
    if (innerRenderer)
        return *innerRenderer;

    auto wrapper = parent.createAnonymousBlock(parent.style().display());
    innerRenderer = wrapper.get();
    m_builder.blockBuilder().attach(parent, WTFMove(wrapper), nullptr);
    parent.setInnerRenderer(*innerRenderer);
    return *innerRenderer;
}

RenderBlock& RenderTreeBuilder::FormControls::findOrCreateParentForChild(RenderMenuList& parent)
{
    auto* innerRenderer = parent.innerRenderer();
    if (innerRenderer)
        return *innerRenderer;

    auto wrapper = parent.createAnonymousBlock();
    innerRenderer = wrapper.get();
    m_builder.blockBuilder().attach(parent, WTFMove(wrapper), nullptr);
    parent.setInnerRenderer(*innerRenderer);
    return *innerRenderer;
}

}
