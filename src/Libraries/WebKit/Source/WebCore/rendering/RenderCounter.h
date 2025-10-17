/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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

#include "CounterContent.h"
#include "RenderText.h"

namespace WebCore {

class CSSCounterStyle;
class CounterNode;

class RenderCounter final : public RenderText {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderCounter);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderCounter);
public:
    RenderCounter(Document&, const CounterContent&);
    virtual ~RenderCounter();

    static void destroyCounterNodes(RenderElement&);
    static void destroyCounterNode(RenderElement&, const AtomString& identifier);
    static void rendererStyleChanged(RenderElement&, const RenderStyle* oldStyle, const RenderStyle& newStyle);

    void updateCounter();
    bool canBeSelectionLeaf() const final { return false; }

private:
    void willBeDestroyed() override;
    static void rendererStyleChangedSlowCase(RenderElement&, const RenderStyle* oldStyle, const RenderStyle& newStyle);
    
    ASCIILiteral renderName() const override;
    String originalText() const override;
    
    RefPtr<CSSCounterStyle> counterStyle() const;

    CounterContent m_counter;
    SingleThreadWeakPtr<CounterNode> m_counterNode;
    SingleThreadWeakPtr<RenderCounter> m_nextForSameCounter;
    friend class CounterNode;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderCounter, isRenderCounter())

#if ENABLE(TREE_DEBUGGING)
// Outside the WebCore namespace for ease of invocation from the debugger.
void showCounterRendererTree(const WebCore::RenderObject*, ASCIILiteral counterName = { });
#endif
