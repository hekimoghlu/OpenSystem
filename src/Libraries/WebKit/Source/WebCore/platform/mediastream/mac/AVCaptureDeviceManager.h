/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#if ENABLE(MEDIA_STREAM) && USE(AVFOUNDATION)

#include "CaptureDeviceManager.h"
#include "RealtimeMediaSource.h"
#include "RealtimeMediaSourceSupportedConstraints.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/RetainPtr.h>
#include <wtf/WorkQueue.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS AVCaptureDevice;
OBJC_CLASS AVCaptureSession;
OBJC_CLASS NSArray;
OBJC_CLASS NSMutableArray;
OBJC_CLASS NSString;
OBJC_CLASS WebCoreAVCaptureDeviceManagerObserver;

namespace WebCore {

class AVCaptureDeviceManager final : public CaptureDeviceManager {
    friend class NeverDestroyed<AVCaptureDeviceManager>;
public:
    static AVCaptureDeviceManager& singleton();

    void refreshCaptureDevices() { refreshCaptureDevicesInternal([] { }, ShouldSetUserPreferredCamera::No); };

private:
    static bool isAvailable();

    AVCaptureDeviceManager();
    ~AVCaptureDeviceManager() final;

    void computeCaptureDevices(CompletionHandler<void()>&&) final;
    const Vector<CaptureDevice>& captureDevices() final;

    void registerForDeviceNotifications();
    void updateCachedAVCaptureDevices();
    Vector<CaptureDevice> retrieveCaptureDevices();
    RetainPtr<NSArray> currentCameras();

    enum class ShouldSetUserPreferredCamera : bool { No, Yes };
    void refreshCaptureDevicesInternal(CompletionHandler<void()>&&, ShouldSetUserPreferredCamera);
    void setUserPreferredCamera();

    RetainPtr<WebCoreAVCaptureDeviceManagerObserver> m_objcObserver;
    Vector<CaptureDevice> m_devices;
    RetainPtr<NSMutableArray> m_avCaptureDevices;
    RetainPtr<NSArray> m_avCaptureDeviceTypes;
    bool m_isInitialized { false };

    Ref<WorkQueue> m_dispatchQueue;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(AVFOUNDATION)
