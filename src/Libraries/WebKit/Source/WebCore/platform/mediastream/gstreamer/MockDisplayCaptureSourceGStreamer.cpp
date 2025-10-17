/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
#include "MockDisplayCaptureSourceGStreamer.h"

namespace WebCore {

CaptureSourceOrError MockDisplayCaptureSourceGStreamer::create(const CaptureDevice& device, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier> pageIdentifier)
{
    auto mockSource = adoptRef(*new MockRealtimeVideoSourceGStreamer(String { device.persistentId() }, AtomString { device.label() }, MediaDeviceHashSalts { hashSalts }, pageIdentifier));

    if (constraints) {
        if (auto error = mockSource->applyConstraints(*constraints))
            return CaptureSourceOrError(CaptureSourceError { error->invalidConstraint });
    }

    Ref<RealtimeMediaSource> source = adoptRef(*new MockDisplayCaptureSourceGStreamer(device, WTFMove(mockSource), WTFMove(hashSalts), pageIdentifier));
    return source;
}

MockDisplayCaptureSourceGStreamer::MockDisplayCaptureSourceGStreamer(const CaptureDevice& device, Ref<MockRealtimeVideoSourceGStreamer>&& source, MediaDeviceHashSalts&& hashSalts, std::optional<PageIdentifier> pageIdentifier)
    : RealtimeVideoCaptureSource(device, WTFMove(hashSalts), pageIdentifier)
    , m_source(WTFMove(source))
    , m_deviceType(device.type())
{
    m_source->addVideoFrameObserver(*this);
}

MockDisplayCaptureSourceGStreamer::~MockDisplayCaptureSourceGStreamer()
{
    m_source->removeVideoFrameObserver(*this);
}

void MockDisplayCaptureSourceGStreamer::stopProducingData()
{
    m_source->removeVideoFrameObserver(*this);
    m_source->stop();
}

void MockDisplayCaptureSourceGStreamer::requestToEnd(RealtimeMediaSourceObserver& callingObserver)
{
    RealtimeMediaSource::requestToEnd(callingObserver);
    m_source->removeVideoFrameObserver(*this);
    m_source->requestToEnd(callingObserver);
}

void MockDisplayCaptureSourceGStreamer::setMuted(bool isMuted)
{
    RealtimeMediaSource::setMuted(isMuted);
    m_source->setMuted(isMuted);
}

void MockDisplayCaptureSourceGStreamer::videoFrameAvailable(VideoFrame& videoFrame, VideoFrameTimeMetadata metadata)
{
    RealtimeMediaSource::videoFrameAvailable(videoFrame, metadata);
}

const RealtimeMediaSourceCapabilities& MockDisplayCaptureSourceGStreamer::capabilities()
{
    if (!m_capabilities) {
        RealtimeMediaSourceCapabilities capabilities(settings().supportedConstraints());

        // FIXME: what should these be?
        // Currently mimicking the values for SCREEN-1 in MockRealtimeMediaSourceCenter.cpp::defaultDevices()
        capabilities.setWidth({ 1, 1920 });
        capabilities.setHeight({ 1, 1080 });
        capabilities.setFrameRate({ .01, 30.0 });

        capabilities.setDeviceId(hashedId());

        m_capabilities = WTFMove(capabilities);
    }
    return m_capabilities.value();
}

const RealtimeMediaSourceSettings& MockDisplayCaptureSourceGStreamer::settings()
{
    if (!m_currentSettings) {
        RealtimeMediaSourceSettings settings;
        settings.setFrameRate(frameRate());

        m_source->ensureIntrinsicSizeMaintainsAspectRatio();
        auto size = m_source->size();
        settings.setWidth(size.width());
        settings.setHeight(size.height());
        settings.setDeviceId(hashedId());
        settings.setDisplaySurface(m_source->mockScreen() ? DisplaySurfaceType::Monitor : DisplaySurfaceType::Window);
        settings.setLogicalSurface(false);

        RealtimeMediaSourceSupportedConstraints supportedConstraints;
        supportedConstraints.setSupportsFrameRate(true);
        supportedConstraints.setSupportsWidth(true);
        supportedConstraints.setSupportsHeight(true);
        supportedConstraints.setSupportsDisplaySurface(true);
        supportedConstraints.setSupportsLogicalSurface(true);
        supportedConstraints.setSupportsDeviceId(true);

        settings.setSupportedConstraints(supportedConstraints);

        m_currentSettings = WTFMove(settings);
    }
    return m_currentSettings.value();
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
