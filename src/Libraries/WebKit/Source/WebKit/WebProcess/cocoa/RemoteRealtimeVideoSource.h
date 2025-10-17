/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "RemoteRealtimeMediaSource.h"
#include "RemoteRealtimeMediaSourceProxy.h"
#include <WebCore/CaptureDevice.h>
#include <WebCore/RealtimeMediaSourceIdentifier.h>

namespace WebCore {
class VideoFrame;
struct MediaConstraints;
}

namespace WebKit {

class UserMediaCaptureManager;

class RemoteRealtimeVideoSource final : public RemoteRealtimeMediaSource {
public:
    static Ref<WebCore::RealtimeMediaSource> create(const WebCore::CaptureDevice&, const WebCore::MediaConstraints*, WebCore::MediaDeviceHashSalts&&, UserMediaCaptureManager&, bool shouldCaptureInGPUProcess, std::optional<WebCore::PageIdentifier>);
    ~RemoteRealtimeVideoSource();

    void remoteVideoFrameAvailable(WebCore::VideoFrame&, WebCore::VideoFrameTimeMetadata);

private:
    RemoteRealtimeVideoSource(WebCore::RealtimeMediaSourceIdentifier, const WebCore::CaptureDevice&, const WebCore::MediaConstraints*, WebCore::MediaDeviceHashSalts&&, UserMediaCaptureManager&, bool shouldCaptureInGPUProcess, std::optional<WebCore::PageIdentifier>);
    RemoteRealtimeVideoSource(RemoteRealtimeMediaSourceProxy&&, WebCore::MediaDeviceHashSalts&&, UserMediaCaptureManager&, std::optional<WebCore::PageIdentifier>);

    Ref<RealtimeMediaSource> clone() final;
    bool setShouldApplyRotation(bool) final;
    double observedFrameRate() const final { return m_observedFrameRate; }

    Deque<double> m_observedFrameTimeStamps;
    double m_observedFrameRate { 0 };
};

} // namespace WebKit

#endif
