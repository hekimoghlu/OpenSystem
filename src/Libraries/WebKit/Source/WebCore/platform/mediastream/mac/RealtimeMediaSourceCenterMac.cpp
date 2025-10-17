/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#include "RealtimeMediaSourceCenter.h"

#if ENABLE(MEDIA_STREAM)

#include "AVCaptureDeviceManager.h"
#include "AVVideoCaptureSource.h"
#include "CoreAudioCaptureSource.h"
#include "DisplayCaptureManagerCocoa.h"
#include "DisplayCaptureSourceCocoa.h"
#include "Logging.h"
#include "MediaStreamPrivate.h"
#include <wtf/MainThread.h>

namespace WebCore {

class VideoCaptureSourceFactoryMac final : public VideoCaptureFactory {
public:
    CaptureSourceOrError createVideoCaptureSource(const CaptureDevice& device, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier> pageIdentifier) final
    {
        ASSERT(device.type() == CaptureDevice::DeviceType::Camera);
#if HAVE(AVCAPTUREDEVICE)
        return AVVideoCaptureSource::create(device, WTFMove(hashSalts), constraints, pageIdentifier);
#else
        UNUSED_PARAM(device);
        UNUSED_PARAM(hashSalts);
        UNUSED_PARAM(constraints);
        UNUSED_PARAM(pageIdentifier);
        return CaptureSourceOrError({ "AVVideoCaptureSource not available"_s , MediaAccessDenialReason::PermissionDenied });
#endif
    }

private:
    CaptureDeviceManager& videoCaptureDeviceManager() { return AVCaptureDeviceManager::singleton(); }
};

class DisplayCaptureSourceFactoryMac final : public DisplayCaptureFactory {
public:
    CaptureSourceOrError createDisplayCaptureSource(const CaptureDevice& device, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier> pageIdentifier) final
    {
        return DisplayCaptureSourceCocoa::create(device, WTFMove(hashSalts), constraints, pageIdentifier);
    }
private:
    DisplayCaptureManager& displayCaptureDeviceManager() { return DisplayCaptureManagerCocoa::singleton(); }
};

AudioCaptureFactory& RealtimeMediaSourceCenter::defaultAudioCaptureFactory()
{
    return CoreAudioCaptureSource::factory();
}

VideoCaptureFactory& RealtimeMediaSourceCenter::defaultVideoCaptureFactory()
{
    static NeverDestroyed<VideoCaptureSourceFactoryMac> factory;
    return factory.get();
}

DisplayCaptureFactory& RealtimeMediaSourceCenter::defaultDisplayCaptureFactory()
{
    static NeverDestroyed<DisplayCaptureSourceFactoryMac> factory;
    return factory.get();
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
