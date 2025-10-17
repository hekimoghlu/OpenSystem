/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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

#include "ScrollbarsController.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>

namespace WebCore {

class ScrollbarsControllerGeneric final : public ScrollbarsController, public CanMakeCheckedPtr<ScrollbarsControllerGeneric> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ScrollbarsControllerGeneric);
public:
    explicit ScrollbarsControllerGeneric(ScrollableArea&);
    virtual ~ScrollbarsControllerGeneric();

private:
    void didAddVerticalScrollbar(Scrollbar*) override;
    void didAddHorizontalScrollbar(Scrollbar*) override;
    void willRemoveVerticalScrollbar(Scrollbar*) override;
    void willRemoveHorizontalScrollbar(Scrollbar*) override;

    void mouseEnteredContentArea() override;
    void mouseExitedContentArea() override;
    void mouseMovedInContentArea() override;
    void contentAreaDidShow() override;
    void contentAreaDidHide() override;
    void notifyContentAreaScrolled(const FloatSize& delta) override;
    void lockOverlayScrollbarStateToHidden(bool) override;

    void overlayScrollbarAnimationTimerFired();
    void showOverlayScrollbars();
    void hideOverlayScrollbars();
    void updateOverlayScrollbarsOpacity();

    Scrollbar* m_horizontalOverlayScrollbar { nullptr };
    Scrollbar* m_verticalOverlayScrollbar { nullptr };
    bool m_overlayScrollbarsLocked { false };
    Timer m_overlayScrollbarAnimationTimer;
    double m_overlayScrollbarAnimationSource { 0 };
    double m_overlayScrollbarAnimationTarget { 0 };
    double m_overlayScrollbarAnimationCurrent { 0 };
    MonotonicTime m_overlayScrollbarAnimationStartTime;
    MonotonicTime m_overlayScrollbarAnimationEndTime;
};

} // namespace WebCore
