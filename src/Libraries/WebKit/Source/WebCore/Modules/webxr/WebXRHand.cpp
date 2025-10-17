/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#include "WebXRHand.h"

#if ENABLE(WEBXR) && ENABLE(WEBXR_HANDS)

#include "WebXRInputSource.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRHand);

Ref<WebXRHand> WebXRHand::create(const WebXRInputSource& inputSource)
{
    return adoptRef(*new WebXRHand(inputSource));
}

WebXRHand::WebXRHand(const WebXRInputSource& inputSource)
    : m_inputSource(inputSource)
{
    RefPtr session = this->session();
    RefPtr document = session ? downcast<Document>(session->scriptExecutionContext()) : nullptr;
    if (!document)
        return;

    size_t jointCount = static_cast<size_t>(XRHandJoint::Count);
    m_joints = Vector<Ref<WebXRJointSpace>>(jointCount, [&](size_t i) {
        return WebXRJointSpace::create(*document, *this, static_cast<XRHandJoint>(i));
    });
}

WebXRHand::~WebXRHand() = default;

RefPtr<WebXRJointSpace> WebXRHand::get(const XRHandJoint& key)
{
    size_t jointIndex = static_cast<size_t>(key);
    if (jointIndex >= m_joints.size())
        return nullptr;

    return m_joints[jointIndex].ptr();
}

WebXRHand::Iterator::Iterator(WebXRHand& hand)
    : m_hand(hand)
{
}

std::optional<KeyValuePair<XRHandJoint, RefPtr<WebXRJointSpace>>> WebXRHand::Iterator::next()
{
    if (m_index >= m_hand->m_joints.size())
        return std::nullopt;

    size_t index = m_index++;
    return KeyValuePair<XRHandJoint, RefPtr<WebXRJointSpace>> { static_cast<XRHandJoint>(index), m_hand->m_joints[index].ptr() };
}

WebXRSession* WebXRHand::session()
{
    if (!m_inputSource)
        return nullptr;

    return m_inputSource->session();
}

void WebXRHand::updateFromInputSource(const PlatformXR::FrameData::InputSource& inputSource)
{
    if (!inputSource.handJoints) {
        m_hasMissingPoses = true;
        return;
    }

    auto& handJoints = *(inputSource.handJoints);
    if (handJoints.size() != m_joints.size()) {
        m_hasMissingPoses = true;
        return;
    }

    bool hasMissingPoses = false;
    for (size_t i = 0; i < handJoints.size(); ++i) {
        if (!handJoints[i])
            hasMissingPoses = true;

        m_joints[i]->updateFromJoint(handJoints[i]);
    }
    m_hasMissingPoses = hasMissingPoses;
}

}

#endif
