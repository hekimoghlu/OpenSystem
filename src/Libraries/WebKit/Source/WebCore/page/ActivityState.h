/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#include <wtf/OptionSet.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

enum class ActivityState : uint16_t {
    WindowIsActive = 1 << 0,
    IsFocused = 1 << 1,
    IsVisible = 1 << 2,
    IsVisibleOrOccluded = 1 << 3,
    IsInWindow = 1 << 4,
    IsVisuallyIdle = 1 << 5,
    IsAudible = 1 << 6,
    IsLoading = 1 << 7,
    IsCapturingMedia = 1 << 8,
    IsConnectedToHardwareConsole = 1 << 9,
};

static constexpr OptionSet<ActivityState> allActivityStates() { return { ActivityState::WindowIsActive, ActivityState::IsFocused, ActivityState::IsVisible, ActivityState::IsVisibleOrOccluded, ActivityState::IsInWindow, ActivityState::IsVisuallyIdle, ActivityState::IsAudible, ActivityState::IsLoading, ActivityState::IsCapturingMedia, ActivityState::IsConnectedToHardwareConsole }; }

enum class ActivityStateForCPUSampling : uint8_t {
    NonVisible,
    VisibleNonActive,
    VisibleAndActive
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, OptionSet<ActivityState>);

} // namespace WebCore
