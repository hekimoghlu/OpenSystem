/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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

#include "AccessibilityMockObject.h"

namespace WebCore {

class RemoteFrame;

class AXRemoteFrame final : public AccessibilityMockObject {
public:
    static Ref<AXRemoteFrame> create(AXID);

#if PLATFORM(COCOA)
    void initializePlatformElementWithRemoteToken(std::span<const uint8_t>, int);
    Vector<uint8_t> generateRemoteToken() const;
    RetainPtr<id> remoteFramePlatformElement() const { return m_remoteFramePlatformElement; }
    pid_t processIdentifier() const { return m_processIdentifier; }
    std::optional<FrameIdentifier> frameID() const { return m_frameID; }
    void setFrameID(FrameIdentifier frameID) { m_frameID = frameID; }
#endif

private:
    virtual ~AXRemoteFrame() = default;
    explicit AXRemoteFrame(AXID);

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::RemoteFrame; }
    bool computeIsIgnored() const final { return false; }
    bool isAXRemoteFrame() const final { return true; }
    LayoutRect elementRect() const final;

#if PLATFORM(COCOA)
    RetainPtr<id> m_remoteFramePlatformElement;
    pid_t m_processIdentifier { 0 };
    std::optional<FrameIdentifier> m_frameID { };
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ACCESSIBILITY(AXRemoteFrame, isAXRemoteFrame())
