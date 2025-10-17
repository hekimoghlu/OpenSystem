/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include "DeviceMotionClient.h"
#include "DeviceMotionController.h"
#include "DeviceMotionData.h"
#include "DeviceOrientationUpdateProvider.h"
#include "MotionManagerClient.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WebCoreMotionManager;

namespace WebCore {

class DeviceMotionClientIOS : public DeviceMotionClient, public MotionManagerClient {
    WTF_MAKE_TZONE_ALLOCATED(DeviceMotionClientIOS);
public:
    DeviceMotionClientIOS(RefPtr<DeviceOrientationUpdateProvider>&&);
    ~DeviceMotionClientIOS() override;
    void setController(DeviceMotionController*) override;
    void startUpdating() override;
    void stopUpdating() override;
    DeviceMotionData* lastMotion() const override;
    void deviceMotionControllerDestroyed() override;

    void motionChanged(double, double, double, double, double, double, std::optional<double>, std::optional<double>, std::optional<double>) override;

private:
    WebCoreMotionManager* m_motionManager { nullptr };
    DeviceMotionController* m_controller { nullptr };
    RefPtr<DeviceMotionData> m_currentDeviceMotionData;
    RefPtr<DeviceOrientationUpdateProvider> m_deviceOrientationUpdateProvider;
    bool m_updating { false };
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
