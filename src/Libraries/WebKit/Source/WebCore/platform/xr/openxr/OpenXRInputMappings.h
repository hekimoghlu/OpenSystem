/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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

#if ENABLE(WEBXR) && USE(OPENXR)

#include <array>
#include <wtf/text/WTFString.h>

namespace PlatformXR {

using OpenXRProfileId = const char* const;
using OpenXRButtonPath = const char* const;

enum class OpenXRButtonType {
    Trigger,
    Squeeze,
    Touchpad,
    Thumbstick,
    Thumbrest,
    ButtonA,
    ButtonB
};

constexpr std::array<OpenXRButtonType, 7> openXRButtonTypes {
    OpenXRButtonType::Trigger, OpenXRButtonType::Squeeze, OpenXRButtonType::Touchpad, OpenXRButtonType::Thumbstick, OpenXRButtonType::Thumbrest,
    OpenXRButtonType::ButtonA, OpenXRButtonType::ButtonB
};

inline String buttonTypeToString(OpenXRButtonType type)
{
    switch (type) {
    case OpenXRButtonType::Trigger: return "trigger"_s;
    case OpenXRButtonType::Squeeze: return "squeeze"_s;
    case OpenXRButtonType::Touchpad: return "touchpad"_s;
    case OpenXRButtonType::Thumbstick: return "thumbstick"_s;
    case OpenXRButtonType::Thumbrest: return "thumbrest"_s;
    case OpenXRButtonType::ButtonA: return "buttona"_s;
    case OpenXRButtonType::ButtonB: return "buttonb"_s;

    default:
        ASSERT_NOT_REACHED();
        return emptyString();
    }
}

struct OpenXRButton {
    OpenXRButtonType type;
    OpenXRButtonPath press { nullptr };
    OpenXRButtonPath touch { nullptr };
    OpenXRButtonPath value { nullptr };
};

enum class OpenXRAxisType {
    Touchpad,
    Thumbstick
};

constexpr std::array<OpenXRAxisType, 2> openXRAxisTypes {
    OpenXRAxisType::Touchpad, OpenXRAxisType::Thumbstick
};

inline String axisTypetoString(OpenXRAxisType type)
{
    switch (type) {
    case OpenXRAxisType::Touchpad: return "touchpad"_s;
    case OpenXRAxisType::Thumbstick: return "thumbstick"_s;
    default:
        ASSERT_NOT_REACHED();
        return emptyString();
    }
}

struct OpenXRAxis {
    OpenXRAxisType type;
    OpenXRButtonPath path;
};

struct OpenXRInputProfile {
    const char* const path { nullptr };
    std::span<const OpenXRProfileId> profileIds;
    std::span<const OpenXRButton> leftButtons;
    std::span<const OpenXRButton> rightButtons;
    std::span<const OpenXRAxis> axes;
};

// https://github.com/immersive-web/webxr-input-profiles/blob/master/packages/registry/profiles/htc/htc-vive.json
constexpr std::array<OpenXRProfileId, 2> HTCViveProfileIds { "htc-vive", "generic-trigger-squeeze-touchpad" };

constexpr std::array<OpenXRButton, 3> HTCViveButtons {
    OpenXRButton { .type = OpenXRButtonType::Trigger, .press = "/input/trigger/click", .value = "/input/trigger/value" },
    OpenXRButton { .type = OpenXRButtonType::Squeeze, .press = "/input/squeeze/click" },
    OpenXRButton { .type = OpenXRButtonType::Touchpad, .press = "/input/trackpad/click", .touch = "/input/trackpad/touch" }
};

constexpr std::array<OpenXRAxis, 1> HTCViveAxes {
    { { OpenXRAxisType::Touchpad, "/input/trackpad" } }
};

constexpr OpenXRInputProfile HTCViveInputProfile {
    "/interaction_profiles/htc/vive_controller",
    HTCViveProfileIds,
    HTCViveButtons,
    HTCViveButtons,
    HTCViveAxes,
};

// Default fallback when there isn't a specific controller binding.
constexpr std::array<const char*, 1> KHRSimpleProfileIds { "generic-button" };

constexpr std::array<OpenXRButton, 1> KHRSimpleButtons {
    OpenXRButton { .type = OpenXRButtonType::Trigger, .press = "/input/select/click" }
};

constexpr OpenXRInputProfile KHRSimpleInputProfile {
    "/interaction_profiles/khr/simple_controller",
    KHRSimpleProfileIds,
    KHRSimpleButtons,
    KHRSimpleButtons,
    { }
};

constexpr std::array<OpenXRInputProfile, 2> openXRInputProfiles { HTCViveInputProfile, KHRSimpleInputProfile };

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
