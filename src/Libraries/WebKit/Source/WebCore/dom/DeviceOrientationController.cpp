/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#include "DeviceOrientationController.h"

#include "DeviceOrientationClient.h"
#include "DeviceOrientationData.h"
#include "DeviceOrientationEvent.h"
#include "EventNames.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceOrientationClient);
WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceOrientationController);

DeviceOrientationController::DeviceOrientationController(DeviceOrientationClient& client)
    : DeviceController(client)
{
    deviceOrientationClient().setController(this);
}

void DeviceOrientationController::didChangeDeviceOrientation(DeviceOrientationData* orientation)
{
    dispatchDeviceEvent(DeviceOrientationEvent::create(eventNames().deviceorientationEvent, orientation));
}

DeviceOrientationClient& DeviceOrientationController::deviceOrientationClient()
{
    return static_cast<DeviceOrientationClient&>(m_client.get());
}

#if PLATFORM(IOS_FAMILY)

// FIXME: We should look to reconcile the iOS and non-iOS differences with this class
// so that we can either remove these functions or remove the PLATFORM(IOS_FAMILY)-guard.

void DeviceOrientationController::suspendUpdates()
{
    m_client->stopUpdating();
}

void DeviceOrientationController::resumeUpdates()
{
    if (!m_listeners.isEmpty())
        m_client->startUpdating();
}

#else

bool DeviceOrientationController::hasLastData()
{
    return deviceOrientationClient().lastOrientation();
}

RefPtr<Event> DeviceOrientationController::getLastEvent()
{
    RefPtr orientation = deviceOrientationClient().lastOrientation();
    return DeviceOrientationEvent::create(eventNames().deviceorientationEvent, orientation.get());
}

#endif // PLATFORM(IOS_FAMILY)

ASCIILiteral DeviceOrientationController::supplementName()
{
    return "DeviceOrientationController"_s;
}

DeviceOrientationController* DeviceOrientationController::from(Page* page)
{
    return static_cast<DeviceOrientationController*>(Supplement<Page>::from(page, supplementName()));
}

bool DeviceOrientationController::isActiveAt(Page* page)
{
    if (DeviceOrientationController* self = DeviceOrientationController::from(page))
        return self->isActive();
    return false;
}

} // namespace WebCore
