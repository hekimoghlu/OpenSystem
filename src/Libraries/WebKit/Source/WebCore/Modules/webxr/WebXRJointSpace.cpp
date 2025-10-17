/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#include "WebXRJointSpace.h"

#if ENABLE(WEBXR) && ENABLE(WEBXR_HANDS)

#include "WebXRFrame.h"
#include "WebXRHand.h"
#include "WebXRRigidTransform.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRJointSpace);

Ref<WebXRJointSpace> WebXRJointSpace::create(Document& document, WebXRHand& hand, XRHandJoint jointName, std::optional<PlatformXR::FrameData::InputSourceHandJoint>&& joint)
{
    return adoptRef(*new WebXRJointSpace(document, hand, jointName, WTFMove(joint)));
}

WebXRJointSpace::WebXRJointSpace(Document& document, WebXRHand& hand, XRHandJoint jointName, std::optional<PlatformXR::FrameData::InputSourceHandJoint>&& joint)
    : WebXRSpace(document, WebXRRigidTransform::create())
    , m_hand(hand)
    , m_jointName(jointName)
    , m_joint(WTFMove(joint))
{
}

WebXRJointSpace::~WebXRJointSpace() = default;

void WebXRJointSpace::updateFromJoint(const std::optional<PlatformXR::FrameData::InputSourceHandJoint>& joint)
{
    m_joint = joint;
}

bool WebXRJointSpace::handHasMissingPoses() const
{
    return !m_hand || m_hand->hasMissingPoses();
}

WebXRSession* WebXRJointSpace::session() const
{
    return m_hand ? m_hand->session() : nullptr;
}

std::optional<TransformationMatrix> WebXRJointSpace::nativeOrigin() const
{
    // https://immersive-web.github.io/webxr-hand-input/#xrjointspace-interface
    // The native origin of the XRJointSpace may only be reported when native origins of
    // all other XRJointSpaces on the same hand are being reported. When a hand is partially
    // obscured the user agent MUST either emulate the obscured joints, or report null poses
    // for all of the joints.
    if (handHasMissingPoses() || !m_joint)
        return std::nullopt;

    return WebXRFrame::matrixFromPose(m_joint->pose.pose);
}

}

#endif
