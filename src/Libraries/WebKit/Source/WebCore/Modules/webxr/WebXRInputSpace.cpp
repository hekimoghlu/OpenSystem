/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#include "WebXRInputSpace.h"

#if ENABLE(WEBXR)

#include "WebXRFrame.h"
#include "WebXRRigidTransform.h"
#include "WebXRSession.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRInputSpace);

// WebXRInputSpace

Ref<WebXRInputSpace> WebXRInputSpace::create(Document& document, WebXRSession& session, const PlatformXR::FrameData::InputSourcePose& pose)
{
    return adoptRef(*new WebXRInputSpace(document, session, pose));
}

WebXRInputSpace::WebXRInputSpace(Document& document, WebXRSession& session, const PlatformXR::FrameData::InputSourcePose& pose)
    : WebXRSpace(document, WebXRRigidTransform::create())
    , m_session(session)
    , m_pose(pose)
{
}

WebXRInputSpace::~WebXRInputSpace() = default;

std::optional<TransformationMatrix> WebXRInputSpace::nativeOrigin() const
{
    return WebXRFrame::matrixFromPose(m_pose.pose);
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
