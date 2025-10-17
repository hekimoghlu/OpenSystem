/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#include "DeviceMotionController.h"

#include "DeviceMotionClient.h"
#include "DeviceMotionData.h"
#include "DeviceMotionEvent.h"
#include "EventNames.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceMotionClient);
WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceMotionController);

DeviceMotionController::DeviceMotionController(DeviceMotionClient& client)
    : DeviceController(client)
{
    deviceMotionClient().setController(this);
}

#if PLATFORM(IOS_FAMILY)

// FIXME: We should look to reconcile the iOS vs. non-iOS differences with this class
// so that we can either remove these functions or the PLATFORM(IOS_FAMILY)-guard.

void DeviceMotionController::suspendUpdates()
{
    m_client->stopUpdating();
}

void DeviceMotionController::resumeUpdates()
{
    if (!m_listeners.isEmpty())
        m_client->startUpdating();
}

#endif
    
void DeviceMotionController::didChangeDeviceMotion(DeviceMotionData* deviceMotionData)
{
    dispatchDeviceEvent(DeviceMotionEvent::create(eventNames().devicemotionEvent, deviceMotionData));
}

DeviceMotionClient& DeviceMotionController::deviceMotionClient()
{
    return downcast<DeviceMotionClient>(m_client.get());
}

bool DeviceMotionController::hasLastData()
{
    return deviceMotionClient().lastMotion();
}

RefPtr<Event> DeviceMotionController::getLastEvent()
{
    RefPtr lastMotion = deviceMotionClient().lastMotion();
    return DeviceMotionEvent::create(eventNames().devicemotionEvent, lastMotion.get());
}

ASCIILiteral DeviceMotionController::supplementName()
{
    return "DeviceMotionController"_s;
}

DeviceMotionController* DeviceMotionController::from(Page* page)
{
    return static_cast<DeviceMotionController*>(Supplement<Page>::from(page, supplementName()));
}

bool DeviceMotionController::isActiveAt(Page* page)
{
    if (DeviceMotionController* self = DeviceMotionController::from(page))
        return self->isActive();
    return false;
}

} // namespace WebCore
