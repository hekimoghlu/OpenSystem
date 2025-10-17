/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderTreeBuilder::Inline {
    WTF_MAKE_TZONE_ALLOCATED(Inline);
public:
    static RenderBoxModelObject& parentCandidateInContinuation(RenderInline& parent, const RenderObject* beforeChild);

    Inline(RenderTreeBuilder&);

    void attach(RenderInline& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void attachIgnoringContinuation(RenderInline& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);

    // Make this private once all the mutation code is in RenderTreeBuilder.
    void childBecameNonInline(RenderInline& parent, RenderElement& child);

private:
    void insertChildToContinuation(RenderInline& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void splitInlines(RenderInline& parent, RenderBlock* fromBlock, RenderBlock* toBlock, RenderBlock* middleBlock, RenderObject* beforeChild, RenderBoxModelObject* oldCont);
    bool newChildIsInline(const RenderInline& parent, const RenderObject& child);
    void splitFlow(RenderInline& parent, RenderObject* beforeChild, RenderPtr<RenderBlock> newBlockBox, RenderPtr<RenderObject> child, RenderBoxModelObject* oldCont);

    RenderTreeBuilder& m_builder;
};

}
