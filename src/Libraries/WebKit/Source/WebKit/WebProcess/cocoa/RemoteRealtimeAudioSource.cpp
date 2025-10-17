/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include "RemoteRealtimeAudioSource.h"

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "GPUProcessConnection.h"
#include "UserMediaCaptureManager.h"
#include "UserMediaCaptureManagerProxyMessages.h"
#include "WebProcess.h"

namespace WebKit {
using namespace WebCore;

Ref<RealtimeMediaSource> RemoteRealtimeAudioSource::create(const CaptureDevice& device, const MediaConstraints* constraints, MediaDeviceHashSalts&& hashSalts, UserMediaCaptureManager& manager, bool shouldCaptureInGPUProcess, std::optional<PageIdentifier> pageIdentifier)
{
    auto source = adoptRef(*new RemoteRealtimeAudioSource(RealtimeMediaSourceIdentifier::generate(), device, constraints, WTFMove(hashSalts), manager, shouldCaptureInGPUProcess, pageIdentifier));
    manager.addSource(source.copyRef());
    manager.remoteCaptureSampleManager().addSource(source.copyRef());
    source->createRemoteMediaSource();
    return source;
}

RemoteRealtimeAudioSource::RemoteRealtimeAudioSource(RealtimeMediaSourceIdentifier identifier, const CaptureDevice& device, const MediaConstraints* constraints, MediaDeviceHashSalts&& hashSalts, UserMediaCaptureManager& manager, bool shouldCaptureInGPUProcess, std::optional<PageIdentifier> pageIdentifier)
    : RemoteRealtimeMediaSource(identifier, device, constraints, WTFMove(hashSalts), manager, shouldCaptureInGPUProcess, pageIdentifier)
{
    ASSERT(device.type() == CaptureDevice::DeviceType::Microphone);
}

RemoteRealtimeAudioSource::~RemoteRealtimeAudioSource() = default;

void RemoteRealtimeAudioSource::remoteAudioSamplesAvailable(const MediaTime& time, const PlatformAudioData& data, const AudioStreamDescription& description, size_t size)
{
    ASSERT(!isMainRunLoop());
    audioSamplesAvailable(time, data, description, size);
}

void RemoteRealtimeAudioSource::setIsInBackground(bool value)
{
    connection().send(Messages::UserMediaCaptureManagerProxy::SetIsInBackground { identifier(), value }, 0);
}

}

#endif
