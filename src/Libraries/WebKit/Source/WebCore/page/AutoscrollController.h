/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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

#include "IntPoint.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>

namespace WebCore {

class EventHandler;
class LocalFrame;
class LocalFrameView;
class Node;
class PlatformMouseEvent;
class RenderBox;
class RenderObject;

enum class AutoscrollType : uint8_t {
    None,
    DragAndDrop,
    Selection,
#if ENABLE(PAN_SCROLLING)
    PanCanStop,
    Pan,
#endif
};

// When the autoscroll or the panScroll is triggered when do the scroll every 50ms to make it smooth.
constexpr Seconds autoscrollInterval { 50_ms };

// AutscrollController handles autoscroll and pan scroll for EventHandler.
class AutoscrollController final : public CanMakeCheckedPtr<AutoscrollController> {
    WTF_MAKE_TZONE_ALLOCATED(AutoscrollController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(AutoscrollController);
public:
    AutoscrollController();
    RenderBox* autoscrollRenderer() const;
    bool autoscrollInProgress() const;
    bool panScrollInProgress() const;
    void startAutoscrollForSelection(RenderObject*);
    void stopAutoscrollTimer(bool rendererIsBeingDestroyed = false);
    void updateAutoscrollRenderer();
    void updateDragAndDrop(Node* targetNode, const IntPoint& eventPosition, WallTime eventTime);
#if ENABLE(PAN_SCROLLING)
    void didPanScrollStart();
    void didPanScrollStop();
    void handleMouseReleaseEvent(const PlatformMouseEvent&);
    void setPanScrollInProgress(bool);
    void startPanScrolling(RenderBox&, const IntPoint&);
#endif

private:
    void autoscrollTimerFired();
    void startAutoscrollTimer();
#if ENABLE(PAN_SCROLLING)
    void updatePanScrollState(LocalFrameView*, const IntPoint&);
#endif

    Timer m_autoscrollTimer;
    SingleThreadWeakPtr<RenderBox> m_autoscrollRenderer;
    AutoscrollType m_autoscrollType { AutoscrollType::None };
    IntPoint m_dragAndDropAutoscrollReferencePosition;
    WallTime m_dragAndDropAutoscrollStartTime;
#if ENABLE(PAN_SCROLLING)
    IntPoint m_panScrollStartPos;
#endif
};

} // namespace WebCore
