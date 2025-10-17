/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include "DeviceController.h"

#include "DeviceClient.h"
#include "Document.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceController);

DeviceController::DeviceController(DeviceClient& client)
    : m_client(client)
    , m_timer(*this, &DeviceController::fireDeviceEvent)
{
}

DeviceController::~DeviceController() = default;

void DeviceController::addDeviceEventListener(LocalDOMWindow& window)
{
    bool wasEmpty = m_listeners.isEmpty();
    m_listeners.add(&window);

    if (hasLastData()) {
        m_lastEventListeners.add(&window);
        if (!m_timer.isActive())
            m_timer.startOneShot(0_s);
    }

    if (wasEmpty)
        m_client->startUpdating();
}

void DeviceController::removeDeviceEventListener(LocalDOMWindow& window)
{
    m_listeners.remove(&window);
    m_lastEventListeners.remove(&window);
    if (m_listeners.isEmpty())
        m_client->stopUpdating();
}

void DeviceController::removeAllDeviceEventListeners(LocalDOMWindow& window)
{
    m_listeners.removeAll(&window);
    m_lastEventListeners.removeAll(&window);
    if (m_listeners.isEmpty())
        m_client->stopUpdating();
}

bool DeviceController::hasDeviceEventListener(LocalDOMWindow& window) const
{
    return m_listeners.contains(&window);
}

void DeviceController::dispatchDeviceEvent(Event& event)
{
    for (auto& listener : copyToVector(m_listeners.values())) {
        RefPtr document = listener->document();
        if (document && !document->activeDOMObjectsAreSuspended() && !document->activeDOMObjectsAreStopped())
            listener->dispatchEvent(event);
    }
}

DeviceClient& DeviceController::client()
{
    return m_client.get();
}

void DeviceController::fireDeviceEvent()
{
    ASSERT(hasLastData());

    m_timer.stop();
    auto listenerVector = copyToVector(m_lastEventListeners.values());
    m_lastEventListeners.clear();
    for (auto& listener : listenerVector) {
        auto document = listener->document();
        if (document && !document->activeDOMObjectsAreSuspended() && !document->activeDOMObjectsAreStopped()) {
            if (RefPtr lastEvent = getLastEvent())
                listener->dispatchEvent(*lastEvent);
        }
    }
}

} // namespace WebCore
