/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#include "ScrollingTreeScrollingNodeDelegate.h"

#include <wtf/TZoneMallocInlines.h>

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingTreeScrollingNode.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeScrollingNodeDelegate);

ScrollingTreeScrollingNodeDelegate::ScrollingTreeScrollingNodeDelegate(ScrollingTreeScrollingNode& scrollingNode)
    : m_scrollingNode(scrollingNode)
{
}

ScrollingTreeScrollingNodeDelegate::~ScrollingTreeScrollingNodeDelegate() = default;

RefPtr<ScrollingTree> ScrollingTreeScrollingNodeDelegate::scrollingTree() const
{
    return protectedScrollingNode()->scrollingTree();
}

FloatPoint ScrollingTreeScrollingNodeDelegate::lastCommittedScrollPosition() const
{
    return m_scrollingNode.lastCommittedScrollPosition();
}

FloatSize ScrollingTreeScrollingNodeDelegate::totalContentsSize()
{
    return m_scrollingNode.totalContentsSize();
}

FloatSize ScrollingTreeScrollingNodeDelegate::reachableContentsSize()
{
    return m_scrollingNode.reachableContentsSize();
}

IntPoint ScrollingTreeScrollingNodeDelegate::scrollOrigin() const
{
    return m_scrollingNode.scrollOrigin();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
