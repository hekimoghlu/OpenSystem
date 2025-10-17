/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "RemoteRealtimeVideoSource.h"

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "UserMediaCaptureManager.h"
#include "UserMediaCaptureManagerProxyMessages.h"

namespace WebKit {
using namespace WebCore;

Ref<RealtimeMediaSource> RemoteRealtimeVideoSource::create(const CaptureDevice& device, const MediaConstraints* constraints, MediaDeviceHashSalts&& hashSalts, UserMediaCaptureManager& manager, bool shouldCaptureInGPUProcess, std::optional<PageIdentifier> pageIdentifier)
{
    auto source = adoptRef(*new RemoteRealtimeVideoSource(RealtimeMediaSourceIdentifier::generate(), device, constraints, WTFMove(hashSalts), manager, shouldCaptureInGPUProcess, pageIdentifier));
    manager.addSource(source.copyRef());
    manager.remoteCaptureSampleManager().addSource(source.copyRef());
    source->createRemoteMediaSource();
    return source;
}

RemoteRealtimeVideoSource::RemoteRealtimeVideoSource(RealtimeMediaSourceIdentifier identifier, const CaptureDevice& device, const MediaConstraints* constraints, MediaDeviceHashSalts&& hashSalts, UserMediaCaptureManager& manager, bool shouldCaptureInGPUProcess, std::optional<PageIdentifier> pageIdentifier)
    : RemoteRealtimeMediaSource(identifier, device, constraints, WTFMove(hashSalts), manager, shouldCaptureInGPUProcess, pageIdentifier)
{
    ASSERT(this->pageIdentifier());
}

RemoteRealtimeVideoSource::RemoteRealtimeVideoSource(RemoteRealtimeMediaSourceProxy&& proxy, MediaDeviceHashSalts&& hashSalts, UserMediaCaptureManager& manager, std::optional<PageIdentifier> pageIdentifier)
    : RemoteRealtimeMediaSource(WTFMove(proxy), WTFMove(hashSalts), manager, pageIdentifier)
{
    ASSERT(this->pageIdentifier());
}

RemoteRealtimeVideoSource::~RemoteRealtimeVideoSource() = default;

bool RemoteRealtimeVideoSource::setShouldApplyRotation(bool shouldApplyRotation)
{
    connection().send(Messages::UserMediaCaptureManagerProxy::SetShouldApplyRotation { identifier(), shouldApplyRotation }, 0);
    return true;
}

Ref<RealtimeMediaSource> RemoteRealtimeVideoSource::clone()
{
    RefPtr<RemoteRealtimeVideoSource> clone;
    callOnMainRunLoopAndWait([this, &clone] {
        if (isEnded() || proxy().isEnded()) {
            clone = this;
            return;
        }

        clone = adoptRef(*new RemoteRealtimeVideoSource(proxy().clone(), MediaDeviceHashSalts { deviceIDHashSalts() }, manager(), *pageIdentifier()));

        clone->m_registerOwnerCallback = m_registerOwnerCallback;
        clone->setSettings(RealtimeMediaSourceSettings { settings() });
        clone->setCapabilities(RealtimeMediaSourceCapabilities { capabilities() });
        clone->setMuted(muted());

        manager().addSource(*clone);
        manager().remoteCaptureSampleManager().addSource(*clone);
        proxy().createRemoteCloneSource(clone->identifier(), *pageIdentifier());

        bool isNewClonedSource = true;
        clone->m_registerOwnerCallback(*clone, isNewClonedSource);
    });

    return clone.releaseNonNull();
}

void RemoteRealtimeVideoSource::remoteVideoFrameAvailable(VideoFrame& frame, VideoFrameTimeMetadata metadata)
{
    MediaTime sampleTime = frame.presentationTime();

    auto frameTime = sampleTime.toDouble();
    m_observedFrameTimeStamps.append(frameTime);
    m_observedFrameTimeStamps.removeAllMatching([&](auto time) {
        return time <= frameTime - 2;
    });

    auto interval = m_observedFrameTimeStamps.last() - m_observedFrameTimeStamps.first();
    if (interval > 1)
        m_observedFrameRate = (m_observedFrameTimeStamps.size() / interval);

    setIntrinsicSize(expandedIntSize(frame.presentationSize()));
    videoFrameAvailable(frame, metadata);
}

}

#endif
