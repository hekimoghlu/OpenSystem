/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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

#if PLATFORM(MAC)

#include "FloatPoint.h"
#include "FloatSize.h"
#include "ScrollAnimator.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Scrollbar;

class ScrollAnimatorMac final : public ScrollAnimator {
    WTF_MAKE_TZONE_ALLOCATED(ScrollAnimatorMac);
public:
    ScrollAnimatorMac(ScrollableArea&);
    virtual ~ScrollAnimatorMac();

private:
    bool handleWheelEvent(const PlatformWheelEvent&) final;

    void handleWheelEventPhase(PlatformWheelEventPhase) final;

    bool isRubberBandInProgress() const final;

    bool processWheelEventForScrollSnap(const PlatformWheelEvent&) final;

    // ScrollingEffectsControllerClient.
    bool allowsHorizontalStretching(const PlatformWheelEvent&) const final;
    bool allowsVerticalStretching(const PlatformWheelEvent&) const final;
    bool shouldRubberBandOnSide(BoxSide) const final;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
