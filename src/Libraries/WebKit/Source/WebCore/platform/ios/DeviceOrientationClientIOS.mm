/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

#import "DeviceOrientationClientIOS.h"

#import "WebCoreMotionManager.h"
#import <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceOrientationClientIOS);

DeviceOrientationClientIOS::DeviceOrientationClientIOS(RefPtr<DeviceOrientationUpdateProvider>&& deviceOrientationUpdateProvider)
    : DeviceOrientationClient()
    , m_deviceOrientationUpdateProvider(WTFMove(deviceOrientationUpdateProvider))
{
}

DeviceOrientationClientIOS::~DeviceOrientationClientIOS()
{
}

void DeviceOrientationClientIOS::setController(DeviceOrientationController* controller)
{
    m_controller = controller;
}

void DeviceOrientationClientIOS::startUpdating()
{
    m_updating = true;

    if (m_deviceOrientationUpdateProvider) {
        m_deviceOrientationUpdateProvider->startUpdatingDeviceOrientation(*this);
        return;
    }

    if (!m_motionManager)
        m_motionManager = [WebCoreMotionManager sharedManager];

    [m_motionManager addOrientationClient:this];
}

void DeviceOrientationClientIOS::stopUpdating()
{
    m_updating = false;

    if (m_deviceOrientationUpdateProvider) {
        m_deviceOrientationUpdateProvider->stopUpdatingDeviceOrientation(*this);
        return;
    }

    // Remove ourselves as the orientation client so we won't get updates.
    [m_motionManager removeOrientationClient:this];
}

DeviceOrientationData* DeviceOrientationClientIOS::lastOrientation() const
{
    return m_currentDeviceOrientation.get();
}

void DeviceOrientationClientIOS::deviceOrientationControllerDestroyed()
{
    if (m_deviceOrientationUpdateProvider) {
        m_deviceOrientationUpdateProvider->stopUpdatingDeviceOrientation(*this);
        return;
    }

    [m_motionManager removeOrientationClient:this];
}
    
void DeviceOrientationClientIOS::orientationChanged(double alpha, double beta, double gamma, double compassHeading, double compassAccuracy)
{
    if (!m_updating)
        return;

#if PLATFORM(IOS_FAMILY_SIMULATOR)
    UNUSED_PARAM(alpha);
    UNUSED_PARAM(beta);
    UNUSED_PARAM(gamma);
    UNUSED_PARAM(compassHeading);
    UNUSED_PARAM(compassAccuracy);
    m_currentDeviceOrientation = DeviceOrientationData::create();
#else
    m_currentDeviceOrientation = DeviceOrientationData::create(alpha, beta, gamma, compassHeading, compassAccuracy);
#endif

    m_controller->didChangeDeviceOrientation(m_currentDeviceOrientation.get());
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
