/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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

#if ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)

#include <WebCore/ScrollingTreeOverflowScrollingNodeMac.h>

namespace WebKit {

class ScrollerPairMac;

class ScrollingTreeOverflowScrollingNodeRemoteMac : public WebCore::ScrollingTreeOverflowScrollingNodeMac {
public:
    static Ref<ScrollingTreeOverflowScrollingNodeRemoteMac> create(WebCore::ScrollingTree&, WebCore::ScrollingNodeID);
    virtual ~ScrollingTreeOverflowScrollingNodeRemoteMac();

private:
    ScrollingTreeOverflowScrollingNodeRemoteMac(WebCore::ScrollingTree&, WebCore::ScrollingNodeID);

    void handleWheelEventPhase(const WebCore::PlatformWheelEventPhase) override;
    void repositionRelatedLayers() override;
    String scrollbarStateForOrientation(WebCore::ScrollbarOrientation) const override;
};

}

#endif // ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)
