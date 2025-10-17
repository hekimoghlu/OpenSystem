/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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

namespace WebKit {

class RemoteRealtimeAudioSource final : public RemoteRealtimeMediaSource {
public:
    static Ref<WebCore::RealtimeMediaSource> create(const WebCore::CaptureDevice&, const WebCore::MediaConstraints*, WebCore::MediaDeviceHashSalts&&, UserMediaCaptureManager&, bool shouldCaptureInGPUProcess, std::optional<WebCore::PageIdentifier>);
    ~RemoteRealtimeAudioSource();

    void remoteAudioSamplesAvailable(const WTF::MediaTime&, const WebCore::PlatformAudioData&, const WebCore::AudioStreamDescription&, size_t);

private:
    RemoteRealtimeAudioSource(WebCore::RealtimeMediaSourceIdentifier, const WebCore::CaptureDevice&, const WebCore::MediaConstraints*, WebCore::MediaDeviceHashSalts&&, UserMediaCaptureManager&, bool shouldCaptureInGPUProcess, std::optional<WebCore::PageIdentifier>);

    void setIsInBackground(bool) final;
};

} // namespace WebKit

#endif
