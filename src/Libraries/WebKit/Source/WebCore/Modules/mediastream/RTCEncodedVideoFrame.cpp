/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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

#if ENABLE(WEB_RTC)
#include "RTCEncodedVideoFrame.h"

#include <JavaScriptCore/ArrayBuffer.h>

namespace WebCore {

RTCEncodedVideoFrame::RTCEncodedVideoFrame(Ref<RTCRtpTransformableFrame>&& frame)
    : RTCEncodedFrame(WTFMove(frame))
    , m_type(m_frame->isKeyFrame() ? Type::Key : Type::Delta)
{
}

RTCEncodedVideoFrame::~RTCEncodedVideoFrame() = default;

uint64_t RTCEncodedVideoFrame::timestamp() const
{
    return m_frame->timestamp();
}

const RTCEncodedVideoFrame::Metadata& RTCEncodedVideoFrame::getMetadata()
{
    if (!m_metadata)
        m_metadata = m_frame->videoMetadata();
    return *m_metadata;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
