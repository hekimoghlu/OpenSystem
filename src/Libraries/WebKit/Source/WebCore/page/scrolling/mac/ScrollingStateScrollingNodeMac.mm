/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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
#import "config.h"
#import "ScrollingStateScrollingNode.h"

#if ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)

#import "GraphicsLayer.h"
#import "Scrollbar.h"
#import "ScrollbarThemeMac.h"
#import "ScrollingStateTree.h"

namespace WebCore {

void ScrollingStateScrollingNode::setScrollerImpsFromScrollbars(Scrollbar* verticalScrollbar, Scrollbar* horizontalScrollbar)
{
    ScrollbarTheme& scrollbarTheme = ScrollbarTheme::theme();
    if (scrollbarTheme.isMockTheme())
        return;
    ScrollbarThemeMac& macTheme = static_cast<ScrollbarThemeMac&>(scrollbarTheme);

    NSScrollerImp *verticalPainter = verticalScrollbar && verticalScrollbar->supportsUpdateOnSecondaryThread()
        ? macTheme.scrollerImpForScrollbar(*verticalScrollbar) : nullptr;
    NSScrollerImp *horizontalPainter = horizontalScrollbar && horizontalScrollbar->supportsUpdateOnSecondaryThread()
        ? macTheme.scrollerImpForScrollbar(*horizontalScrollbar) : nullptr;

    if (m_verticalScrollerImp == verticalPainter && m_horizontalScrollerImp == horizontalPainter)
        return;

    m_verticalScrollerImp = verticalPainter;
    m_horizontalScrollerImp = horizontalPainter;

    setPropertyChanged(Property::PainterForScrollbar);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)
