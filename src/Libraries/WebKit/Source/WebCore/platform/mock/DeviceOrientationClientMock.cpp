/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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
#include "DeviceOrientationClientMock.h"

#include "DeviceOrientationController.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceOrientationClientMock);

DeviceOrientationClientMock::DeviceOrientationClientMock()
    : m_controller(0)
    , m_timer(*this, &DeviceOrientationClientMock::timerFired)
    , m_isUpdating(false)
{
}

void DeviceOrientationClientMock::setController(DeviceOrientationController* controller)
{
    ASSERT(!m_controller);
    m_controller = controller;
    ASSERT(m_controller);
}

void DeviceOrientationClientMock::startUpdating()
{
    m_isUpdating = true;
}

void DeviceOrientationClientMock::stopUpdating()
{
    m_isUpdating = false;
    m_timer.stop();
}

void DeviceOrientationClientMock::setOrientation(RefPtr<DeviceOrientationData>&& orientation)
{
    m_orientation = WTFMove(orientation);
    if (m_isUpdating && !m_timer.isActive())
        m_timer.startOneShot(0_s);
}

void DeviceOrientationClientMock::timerFired()
{
    m_timer.stop();
    m_controller->didChangeDeviceOrientation(m_orientation.get());
}

} // namespace WebCore
