/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#include "NodeRareData.h"

#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NodeMutationObserverData);

struct SameSizeAsNodeRareData {
    void* m_pointer[2];
    WeakPtr<Node, WeakPtrImplWithEventTargetData> m_weakPointer;
    bool m_isElementRareData;
};

static_assert(sizeof(NodeRareData) == sizeof(SameSizeAsNodeRareData), "NodeRareData should stay small");

// Ensure the 10 bits reserved for the m_connectedFrameCount cannot overflow
static_assert(Page::maxNumberOfFrames < 1024, "Frame limit should fit in rare data count");

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(NodeListsNodeData);
DEFINE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER(NodeRareData);

} // namespace WebCore
