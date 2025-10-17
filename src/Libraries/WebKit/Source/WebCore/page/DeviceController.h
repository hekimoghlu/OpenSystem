/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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

#include "Event.h"
#include "LocalDOMWindow.h"
#include "Supplementable.h"
#include "Timer.h"
#include <wtf/CheckedRef.h>
#include <wtf/HashCountedSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DeviceClient;
class Page;

class DeviceController : public Supplement<Page>, public CanMakeCheckedPtr<DeviceController> {
    WTF_MAKE_TZONE_ALLOCATED(DeviceController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DeviceController);
public:
    explicit DeviceController(DeviceClient&);
    virtual ~DeviceController();

    void addDeviceEventListener(LocalDOMWindow&);
    void removeDeviceEventListener(LocalDOMWindow&);
    void removeAllDeviceEventListeners(LocalDOMWindow&);
    bool hasDeviceEventListener(LocalDOMWindow&) const;

    void dispatchDeviceEvent(Event&);
    bool isActive() { return !m_listeners.isEmpty(); }
    DeviceClient& client();

    virtual bool hasLastData() { return false; }
    virtual RefPtr<Event> getLastEvent() { return nullptr; }

protected:
    void fireDeviceEvent();

    HashCountedSet<RefPtr<LocalDOMWindow>> m_listeners;
    HashCountedSet<RefPtr<LocalDOMWindow>> m_lastEventListeners;
    WeakRef<DeviceClient> m_client;
    Timer m_timer;
};

} // namespace WebCore
