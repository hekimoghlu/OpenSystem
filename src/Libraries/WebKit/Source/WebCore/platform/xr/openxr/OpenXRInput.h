/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

class OpenXRInputSource;

class OpenXRInput {
    WTF_MAKE_TZONE_ALLOCATED(OpenXRInput);
    WTF_MAKE_NONCOPYABLE(OpenXRInput);
public:
    static std::unique_ptr<OpenXRInput> create(XrInstance, XrSession, XrSpace);

    Vector<FrameData::InputSource> collectInputSources(const XrFrameState&) const;
    void updateInteractionProfile();

private:
    OpenXRInput(XrInstance, XrSession, XrSpace);
    XrResult initialize();

    XrInstance m_instance { XR_NULL_HANDLE };
    XrSession m_session { XR_NULL_HANDLE }; 
    XrSpace m_localSpace { XR_NULL_HANDLE };
    Vector<UniqueRef<OpenXRInputSource>> m_inputSources;
    InputSourceHandle m_handleIndex { 0 };
};

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
