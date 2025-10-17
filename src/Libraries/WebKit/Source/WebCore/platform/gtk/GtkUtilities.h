/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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

#include "DragActions.h"
#include <gtk/gtk.h>
#include <wtf/MonotonicTime.h>
#include <wtf/WallTime.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class IntPoint;

IntPoint convertWidgetPointToScreenPoint(GtkWidget*, const IntPoint&);
bool widgetIsOnscreenToplevelWindow(GtkWidget*);
IntPoint widgetRootCoords(GtkWidget*, int, int);
void widgetDevicePosition(GtkWidget*, GdkDevice*, double*, double*, GdkModifierType*);
unsigned widgetKeyvalToKeycode(GtkWidget*, unsigned);

template<typename GdkEventType>
WallTime wallTimeForEvent(const GdkEventType* event)
{
    const auto eventTime = gdk_event_get_time(reinterpret_cast<GdkEvent*>(const_cast<GdkEventType*>(event)));
    if (eventTime == GDK_CURRENT_TIME)
        return WallTime::now();
    return MonotonicTime::fromRawSeconds(eventTime / 1000.).approximateWallTime();
}

template<>
WallTime wallTimeForEvent(const GdkEvent*);

WEBCORE_EXPORT unsigned stateModifierForGdkButton(unsigned button);

WEBCORE_EXPORT OptionSet<DragOperation> gdkDragActionToDragOperation(GdkDragAction);
WEBCORE_EXPORT GdkDragAction dragOperationToGdkDragActions(OptionSet<DragOperation>);
WEBCORE_EXPORT GdkDragAction dragOperationToSingleGdkDragAction(OptionSet<DragOperation>);

void monitorWorkArea(GdkMonitor*, GdkRectangle*);

bool shouldUseOverlayScrollbars();

WEBCORE_EXPORT bool eventModifiersContainCapsLock(GdkEvent*);

} // namespace WebCore
