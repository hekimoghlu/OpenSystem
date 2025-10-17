/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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

#include "OpenXRInputMappings.h"
#include "OpenXRUtils.h"

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace PlatformXR {

class OpenXRInputSource {
    WTF_MAKE_TZONE_ALLOCATED(OpenXRInputSource);
    WTF_MAKE_NONCOPYABLE(OpenXRInputSource);
public:
    using SuggestedBindings = UncheckedKeyHashMap<const char*, Vector<XrActionSuggestedBinding>>;
    static std::unique_ptr<OpenXRInputSource> create(XrInstance, XrSession, XRHandedness, InputSourceHandle);
    ~OpenXRInputSource();

    XrResult suggestBindings(SuggestedBindings&) const;
    std::optional<FrameData::InputSource> getInputSource(XrSpace, const XrFrameState&) const;
    XrActionSet actionSet() const { return m_actionSet; }
    XrResult updateInteractionProfile();

private:
    OpenXRInputSource(XrInstance, XrSession, XRHandedness, InputSourceHandle);

    struct OpenXRButtonActions {
        XrAction press { XR_NULL_HANDLE };
        XrAction touch { XR_NULL_HANDLE };
        XrAction value { XR_NULL_HANDLE };
    };

    XrResult initialize();
    XrResult createSpaceAction(XrAction, XrSpace&) const;
    XrResult createAction(XrActionType, const String& name, XrAction&) const;
    XrResult createButtonActions(OpenXRButtonType, const String& prefix, OpenXRButtonActions&) const;
    XrResult createBinding(const char* profilePath, XrAction, const String& bindingPath, SuggestedBindings&) const;

    XrResult getPose(XrSpace, XrSpace, const XrFrameState&, FrameData::InputSourcePose&) const;
    std::optional<FrameData::InputSourceButton> getButton(OpenXRButtonType) const;
    std::optional<XrVector2f> getAxis(OpenXRAxisType) const;
    XrResult getActionState(XrAction, bool*) const;
    XrResult getActionState(XrAction, float*) const;
    XrResult getActionState(XrAction, XrVector2f*) const;

    XrInstance m_instance { XR_NULL_HANDLE };
    XrSession m_session { XR_NULL_HANDLE };
    XRHandedness m_handedness { XRHandedness::Left };
    InputSourceHandle m_handle { 0 };
    String m_subactionPathName;
    XrPath m_subactionPath { XR_NULL_PATH };
    XrActionSet m_actionSet { XR_NULL_HANDLE };
    XrAction m_gripAction { XR_NULL_HANDLE };
    XrSpace m_gripSpace { XR_NULL_HANDLE };
    XrAction m_pointerAction { XR_NULL_HANDLE };
    XrSpace m_pointerSpace { XR_NULL_HANDLE };
    using OpenXRButtonActionsMap = UncheckedKeyHashMap<OpenXRButtonType, OpenXRButtonActions, IntHash<OpenXRButtonType>, WTF::StrongEnumHashTraits<OpenXRButtonType>>;
    OpenXRButtonActionsMap m_buttonActions;
    using OpenXRAxesMap = UncheckedKeyHashMap<OpenXRAxisType, XrAction, IntHash<OpenXRAxisType>, WTF::StrongEnumHashTraits<OpenXRAxisType>>;
    OpenXRAxesMap m_axisActions;
    Vector<String> m_profiles;
};

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
