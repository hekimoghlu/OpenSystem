/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#include "ActivityState.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, OptionSet<ActivityState> flags)
{
    bool didAppend = false;

    auto appendIf = [&](auto flag, auto message) {
        if (!flags.contains(flag))
            return;
        if (didAppend)
            ts << ", ";
        ts << message;
        didAppend = true;
    };
    
    appendIf(ActivityState::WindowIsActive, "active window");
    appendIf(ActivityState::IsFocused, "focused");
    appendIf(ActivityState::IsVisible, "visible");
    appendIf(ActivityState::IsVisibleOrOccluded, "visible or occluded");
    appendIf(ActivityState::IsInWindow, "in-window");
    appendIf(ActivityState::IsVisuallyIdle, "visually idle");
    appendIf(ActivityState::IsAudible, "audible");
    appendIf(ActivityState::IsLoading, "loading");
    appendIf(ActivityState::IsCapturingMedia, "capturing media");
    appendIf(ActivityState::IsConnectedToHardwareConsole, "attached to hardware console");

    return ts;
}

} // namespace WebCore
