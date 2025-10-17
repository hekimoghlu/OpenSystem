/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollTypes.h"
#include <wtf/Markable.h>

namespace WebCore {

class PlatformWheelEvent;
class ScrollingTree;

// This class tracks state related to MayBegin, Begin, Ended and Canceled wheel events, which
// should propagate to the main thread in order to update overlay scrollbars.
class ScrollingTreeGestureState {
public:
    ScrollingTreeGestureState(ScrollingTree&);

    void receivedWheelEvent(const PlatformWheelEvent&);

    bool handleGestureCancel(const PlatformWheelEvent&);

    void nodeDidHandleEvent(ScrollingNodeID, const PlatformWheelEvent&);
    
private:
    void clearAllNodes();

    ScrollingTree& m_scrollingTree;
    Markable<ScrollingNodeID> m_mayBeginNodeID;
    Markable<ScrollingNodeID> m_activeNodeID;
};

}

#endif // ENABLE(ASYNC_SCROLLING)
