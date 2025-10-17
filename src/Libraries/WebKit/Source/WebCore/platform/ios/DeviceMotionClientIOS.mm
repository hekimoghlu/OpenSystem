/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#import "DeviceMotionClientIOS.h"

#import "WebCoreMotionManager.h"
#import <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceMotionClientIOS);

DeviceMotionClientIOS::DeviceMotionClientIOS(RefPtr<DeviceOrientationUpdateProvider>&& deviceOrientationUpdateProvider)
    : DeviceMotionClient()
    , m_deviceOrientationUpdateProvider(WTFMove(deviceOrientationUpdateProvider))
{
}

DeviceMotionClientIOS::~DeviceMotionClientIOS()
{
}

void DeviceMotionClientIOS::setController(DeviceMotionController* controller)
{
    m_controller = controller;
}

void DeviceMotionClientIOS::startUpdating()
{
    m_updating = true;

    if (m_deviceOrientationUpdateProvider) {
        m_deviceOrientationUpdateProvider->startUpdatingDeviceMotion(*this);
        return;
    }

    if (!m_motionManager)
        m_motionManager = [WebCoreMotionManager sharedManager];

    [m_motionManager addMotionClient:this];
}

void DeviceMotionClientIOS::stopUpdating()
{
    m_updating = false;

    if (m_deviceOrientationUpdateProvider) {
        m_deviceOrientationUpdateProvider->stopUpdatingDeviceMotion(*this);
        return;
    }

    // Remove ourselves as the motion client so we won't get updates.
    [m_motionManager removeMotionClient:this];
}

DeviceMotionData* DeviceMotionClientIOS::lastMotion() const
{
    return m_currentDeviceMotionData.get();
}

void DeviceMotionClientIOS::deviceMotionControllerDestroyed()
{
    if (m_deviceOrientationUpdateProvider) {
        m_deviceOrientationUpdateProvider->stopUpdatingDeviceMotion(*this);
        return;
    }

    [m_motionManager removeMotionClient:this];
}

void DeviceMotionClientIOS::motionChanged(double xAcceleration, double yAcceleration, double zAcceleration, double xAccelerationIncludingGravity, double yAccelerationIncludingGravity, double zAccelerationIncludingGravity, std::optional<double> xRotationRate, std::optional<double> yRotationRate, std::optional<double> zRotationRate)
{
    if (!m_updating)
        return;

#if PLATFORM(IOS_FAMILY_SIMULATOR)
    UNUSED_PARAM(xAcceleration);
    UNUSED_PARAM(yAcceleration);
    UNUSED_PARAM(zAcceleration);
    UNUSED_PARAM(xAccelerationIncludingGravity);
    UNUSED_PARAM(yAccelerationIncludingGravity);
    UNUSED_PARAM(zAccelerationIncludingGravity);
    UNUSED_PARAM(xRotationRate);
    UNUSED_PARAM(yRotationRate);
    UNUSED_PARAM(zRotationRate);

    auto accelerationIncludingGravity = DeviceMotionData::Acceleration::create();
    auto acceleration = DeviceMotionData::Acceleration::create();
    auto rotationRate = DeviceMotionData::RotationRate::create();
#else
    auto accelerationIncludingGravity = DeviceMotionData::Acceleration::create(xAccelerationIncludingGravity, yAccelerationIncludingGravity, zAccelerationIncludingGravity);

    RefPtr<DeviceMotionData::Acceleration> acceleration = DeviceMotionData::Acceleration::create(xAcceleration, yAcceleration, zAcceleration);
    RefPtr<DeviceMotionData::RotationRate> rotationRate;
    // Not all devices have a gyroscope.
    if (xRotationRate && yRotationRate && zRotationRate)
        rotationRate = DeviceMotionData::RotationRate::create(*xRotationRate, *yRotationRate, *zRotationRate);
#endif // PLATFORM(IOS_FAMILY_SIMULATOR)

    m_currentDeviceMotionData = DeviceMotionData::create(WTFMove(acceleration), WTFMove(accelerationIncludingGravity), WTFMove(rotationRate), kMotionUpdateInterval);
    m_controller->didChangeDeviceMotion(m_currentDeviceMotionData.get());
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
