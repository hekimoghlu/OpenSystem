/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#if ENABLE(WEBXR) && ENABLE(WEBXR_HANDS)

#include "Document.h"
#include "PlatformXR.h"
#include "WebXRSpace.h"
#include "XRHandJoint.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class WebXRHand;
class WebXRSession;

class WebXRJointSpace final: public RefCounted<WebXRJointSpace>, public WebXRSpace {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRJointSpace);
public:
    static Ref<WebXRJointSpace> create(Document&, WebXRHand&, XRHandJoint, std::optional<PlatformXR::FrameData::InputSourceHandJoint>&& = std::nullopt);
    virtual ~WebXRJointSpace();

    using RefCounted<WebXRJointSpace>::ref;
    using RefCounted<WebXRJointSpace>::deref;

    XRHandJoint jointName() const { return m_jointName; }

    float radius() const { return m_joint ? m_joint->radius : 0; }
    void updateFromJoint(const std::optional<PlatformXR::FrameData::InputSourceHandJoint>&);
    bool handHasMissingPoses() const;

    WebXRSession* session() const final;

private:
    WebXRJointSpace(Document&, WebXRHand&, XRHandJoint, std::optional<PlatformXR::FrameData::InputSourceHandJoint>&&);

    std::optional<TransformationMatrix> nativeOrigin() const final;

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    bool isJointSpace() const final { return true; }

    WeakPtr<WebXRHand> m_hand;
    XRHandJoint m_jointName { XRHandJoint::Wrist };
    std::optional<PlatformXR::FrameData::InputSourceHandJoint> m_joint;
};

}

SPECIALIZE_TYPE_TRAITS_WEBXRSPACE(WebXRJointSpace, isJointSpace())

#endif
