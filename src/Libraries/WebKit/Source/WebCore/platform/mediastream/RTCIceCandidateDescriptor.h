/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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

#if ENABLE(WEB_RTC)

#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCIceCandidateDescriptor : public RefCounted<RTCIceCandidateDescriptor> {
public:
    static Ref<RTCIceCandidateDescriptor> create(const String& candidate, const String& sdpMid, unsigned short sdpMLineIndex);
    virtual ~RTCIceCandidateDescriptor();

    const String& candidate() const { return m_candidate; }
    const String& sdpMid() const { return m_sdpMid; }
    unsigned short sdpMLineIndex() const { return m_sdpMLineIndex; }

private:
    RTCIceCandidateDescriptor(const String& candidate, const String& sdpMid, unsigned short sdpMLineIndex);

    String m_candidate;
    String m_sdpMid;
    unsigned short m_sdpMLineIndex;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
