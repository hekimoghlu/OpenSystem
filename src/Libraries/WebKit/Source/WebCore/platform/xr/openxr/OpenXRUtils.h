/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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

// EGL symbols required by openxr_platform.h
#if USE(LIBEPOXY)
#define __GBM__ 1
#include <epoxy/egl.h>
#else
#include <EGL/egl.h>
#endif

#include "Logging.h"
#include "PlatformXR.h"
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace PlatformXR {

template<typename T, XrStructureType StructureType>
T createStructure()
{
    T object;
    std::memset(&object, 0, sizeof(T));
    object.type = StructureType;
    object.next = nullptr;
    return object;
}

inline String resultToString(XrResult value, XrInstance instance)
{
    char buffer[XR_MAX_RESULT_STRING_SIZE];
    XrResult result = xrResultToString(instance, value, buffer);
    if (result == XR_SUCCESS)
        return String::fromLatin1(buffer);
    return makeString("<unknown "_s, int(value), '>');
}

#define RETURN_IF_FAILED(call, label, instance, ...)                                                      \
    {                                                                                                     \
        auto xrResult = call;                                                                             \
        if (XR_FAILED(xrResult)) {                                                                        \
            LOG(XR, "%s %s: %s\n", __func__, label, resultToString(xrResult, instance).utf8().data());    \
            return __VA_ARGS__;                                                                           \
        }                                                                                                 \
    }

#define RETURN_RESULT_IF_FAILED(call, instance, ...)                                                                  \
    {                                                                                                                 \
        auto xrResult = call;                                                                                         \
        if (XR_FAILED(xrResult)) {                                                                                    \
            LOG(XR, "%s %s: %s\n", __func__, #call, resultToString(xrResult, instance).utf8().data());                \
            return xrResult;                                                                                          \
        }                                                                                                             \
    }

#define LOG_IF_FAILED(result, call, instance, ...)                                           \
    if (XR_FAILED(result))                                                                   \
        LOG(XR, "%s %s: %s\n", __func__, call, resultToString(result, instance).utf8().data());


inline FrameData::Pose XrPosefToPose(XrPosef pose)
{
    FrameData::Pose result;
    result.orientation = { pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w };
    result.position = { pose.position.x, pose.position.y, pose.position.z };
    return result;
}

inline FrameData::View xrViewToPose(XrView view)
{
    FrameData::View pose;
    pose.projection = FrameData::Fov { std::abs(view.fov.angleUp), std::abs(view.fov.angleDown), std::abs(view.fov.angleLeft), std::abs(view.fov.angleRight) };
    pose.offset = XrPosefToPose(view.pose);
    return pose;
}

inline XrPosef XrPoseIdentity()
{
    XrPosef pose;
    pose.orientation.w = 1.0;
    return pose;
}

inline XrViewConfigurationType toXrViewConfigurationType(SessionMode mode)
{
    switch (mode) {
    case SessionMode::ImmersiveVr:
        return XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    case SessionMode::Inline:
    case SessionMode::ImmersiveAr:
        return XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO;
    };
    ASSERT_NOT_REACHED();
    return XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO;
}

inline String handednessToString(XRHandedness handedness)
{
    switch (handedness) {
    case XRHandedness::Left:
        return "left"_s;
    case XRHandedness::Right:
        return "right"_s;
    default:
        ASSERT_NOT_REACHED();
        return emptyString();
    }
}

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
