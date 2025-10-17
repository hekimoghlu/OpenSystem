/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

#include "PlatformXR.h"
#include "WebXRInputSource.h"
#include "WebXRJointSpace.h"
#include "WebXRSession.h"
#include "XRHandJoint.h"
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class WebXRHand : public RefCountedAndCanMakeWeakPtr<WebXRHand> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRHand);
public:

    static Ref<WebXRHand> create(const WebXRInputSource&);
    ~WebXRHand();

    unsigned size() const { return m_joints.size(); }

    RefPtr<WebXRJointSpace> get(const XRHandJoint& key);

    class Iterator {
    public:
        explicit Iterator(WebXRHand&);
        std::optional<KeyValuePair<XRHandJoint, RefPtr<WebXRJointSpace>>> next();

    private:
        Ref<WebXRHand> m_hand;
        size_t m_index { 0 };
    };
    Iterator createIterator(ScriptExecutionContext*) { return Iterator(*this); }

    // For GC reachability.
    WebXRSession* session();

    bool hasMissingPoses() const { return m_hasMissingPoses; }
    void updateFromInputSource(const PlatformXR::FrameData::InputSource&);

private:
    WebXRHand(const WebXRInputSource&);

    FixedVector<Ref<WebXRJointSpace>> m_joints;
    bool m_hasMissingPoses { true };
    WeakPtr<WebXRInputSource> m_inputSource;
};

}
#endif
