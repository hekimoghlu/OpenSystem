/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

#if PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)

#include "DeviceOrientationClient.h"
#include "DeviceOrientationController.h"
#include "DeviceOrientationData.h"
#include "DeviceOrientationUpdateProvider.h"
#include "MotionManagerClient.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WebCoreMotionManager;

namespace WebCore {

class DeviceOrientationClientIOS : public DeviceOrientationClient, public MotionManagerClient {
    WTF_MAKE_TZONE_ALLOCATED(DeviceOrientationClientIOS);
public:
    DeviceOrientationClientIOS(RefPtr<DeviceOrientationUpdateProvider>&&);
    ~DeviceOrientationClientIOS() override;
    void setController(DeviceOrientationController*) override;
    void startUpdating() override;
    void stopUpdating() override;
    DeviceOrientationData* lastOrientation() const override;
    void deviceOrientationControllerDestroyed() override;

    void orientationChanged(double, double, double, double, double) override;

private:
    WebCoreMotionManager* m_motionManager  { nullptr };
    DeviceOrientationController* m_controller  { nullptr };
    RefPtr<DeviceOrientationData> m_currentDeviceOrientation;
    RefPtr<DeviceOrientationUpdateProvider> m_deviceOrientationUpdateProvider;
    bool m_updating { false };
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
