/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#import "config.h"
#import "MockRealtimeVideoSourceMac.h"

#if ENABLE(MEDIA_STREAM)

#import "GraphicsContextCG.h"
#import "ImageBuffer.h"
#import "ImageTransferSessionVT.h"
#import "MediaConstraints.h"
#import "MockRealtimeMediaSourceCenter.h"
#import "NotImplemented.h"
#import "PlatformLayer.h"
#import "RealtimeMediaSourceSettings.h"
#import "RealtimeVideoUtilities.h"
#import "VideoFrame.h"
#import <QuartzCore/CALayer.h>
#import <QuartzCore/CATransaction.h>
#import <objc/runtime.h>
#import <wtf/MediaTime.h>

#import "CoreVideoSoftLink.h"
#import <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

CaptureSourceOrError MockRealtimeVideoSource::create(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier> pageIdentifier)
{
#ifndef NDEBUG
    auto device = MockRealtimeMediaSourceCenter::mockDeviceWithPersistentID(deviceID);
    ASSERT(device);
    if (!device)
        return CaptureSourceOrError({ "No mock camera device"_s , MediaAccessDenialReason::PermissionDenied });
#endif

    Ref<RealtimeMediaSource> source = MockRealtimeVideoSourceMac::create(WTFMove(deviceID), WTFMove(name), WTFMove(hashSalts), pageIdentifier);
    if (constraints) {
        if (auto error = source->applyConstraints(*constraints))
            return CaptureSourceOrError(CaptureSourceError { error->invalidConstraint });
    }

    return source;
}

Ref<MockRealtimeVideoSource> MockRealtimeVideoSourceMac::createForMockDisplayCapturer(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&& hashSalts, std::optional<PageIdentifier> pageIdentifier)
{
    return adoptRef(*new MockRealtimeVideoSourceMac(WTFMove(deviceID), WTFMove(name), WTFMove(hashSalts), pageIdentifier));
}

MockRealtimeVideoSourceMac::MockRealtimeVideoSourceMac(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&& hashSalts, std::optional<PageIdentifier> pageIdentifier)
    : MockRealtimeVideoSource(WTFMove(deviceID), WTFMove(name), WTFMove(hashSalts), pageIdentifier)
{
}

MockRealtimeVideoSourceMac::~MockRealtimeVideoSourceMac() = default;

void MockRealtimeVideoSourceMac::updateSampleBuffer()
{
    ASSERT(!isMainThread());

    RefPtr imageBuffer = this->imageBufferInternal();
    if (!imageBuffer)
        return;

    if (!m_imageTransferSession) {
        m_imageTransferSession = ImageTransferSessionVT::create(preferedPixelBufferFormat());
        m_imageTransferSession->setMaximumBufferPoolSize(10);
    }

    PlatformImagePtr platformImage;
    if (auto nativeImage = imageBuffer->copyNativeImage())
        platformImage = nativeImage->platformImage();

    auto presentationTime = MediaTime::createWithDouble((elapsedTime() + 100_ms).seconds());
    auto videoFrame = m_imageTransferSession->createVideoFrame(platformImage.get(), presentationTime, size(), videoFrameRotation());
    if (!videoFrame) {
        static const size_t MaxPixelGenerationFailureCount = 150;
        if (++m_pixelGenerationFailureCount > MaxPixelGenerationFailureCount) {
            callOnMainThread([protectedThis = Ref { *this }] {
                protectedThis->captureFailed();
            });
        }
        return;
    }

    m_pixelGenerationFailureCount = 0;
    VideoFrameTimeMetadata metadata;
    metadata.captureTime = MonotonicTime::now().secondsSinceEpoch();
    dispatchVideoFrameToObservers(*videoFrame, metadata);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
