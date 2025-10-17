/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

class RenderBlockFlow;
class RenderMultiColumnFlow;

class RenderTreeBuilder::MultiColumn {
    WTF_MAKE_TZONE_ALLOCATED(MultiColumn);
public:
    MultiColumn(RenderTreeBuilder&);

    void updateAfterDescendants(RenderBlockFlow&);
    // Some renderers (column spanners) are moved out of the flow thread to live among column
    // sets. If |child| is such a renderer, resolve it to the placeholder that lives at the original
    // location in the tree.
    RenderObject* resolveMovedChild(RenderFragmentedFlow& enclosingFragmentedFlow, RenderObject* beforeChild);
    void restoreColumnSpannersForContainer(RenderMultiColumnFlow&, const RenderElement& container);
    void multiColumnDescendantInserted(RenderMultiColumnFlow&, RenderObject& newDescendant);
    void multiColumnRelativeWillBeRemoved(RenderMultiColumnFlow&, RenderObject& relative, RenderTreeBuilder::CanCollapseAnonymousBlock);
    static RenderObject* adjustBeforeChildForMultiColumnSpannerIfNeeded(RenderObject& beforeChild);

private:
    void createFragmentedFlow(RenderBlockFlow&);
    void destroyFragmentedFlow(RenderBlockFlow&);
    RenderObject* processPossibleSpannerDescendant(RenderMultiColumnFlow&, RenderObject*& subtreeRoot, RenderObject& descendant);
    void handleSpannerRemoval(RenderMultiColumnFlow&, RenderObject& spanner, RenderTreeBuilder::CanCollapseAnonymousBlock);

    RenderTreeBuilder& m_builder;
};

}
