/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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

#include "DeviceOrientationOrMotionPermissionState.h"
#include "Event.h"
#include "IDLTypes.h"

namespace WebCore {

class DeviceOrientationData;
class Document;
template<typename IDLType> class DOMPromiseDeferred;

class DeviceOrientationEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DeviceOrientationEvent);
public:
    static Ref<DeviceOrientationEvent> create(const AtomString& eventType, DeviceOrientationData* orientation)
    {
        return adoptRef(*new DeviceOrientationEvent(eventType, orientation));
    }

    static Ref<DeviceOrientationEvent> createForBindings()
    {
        return adoptRef(*new DeviceOrientationEvent);
    }

    virtual ~DeviceOrientationEvent();

    std::optional<double> alpha() const;
    std::optional<double> beta() const;
    std::optional<double> gamma() const;

#if PLATFORM(IOS_FAMILY)
    std::optional<double> compassHeading() const;
    std::optional<double> compassAccuracy() const;

    void initDeviceOrientationEvent(const AtomString& type, bool bubbles, bool cancelable, std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<double> compassHeading, std::optional<double> compassAccuracy);
#else
    std::optional<bool> absolute() const;

    void initDeviceOrientationEvent(const AtomString& type, bool bubbles, bool cancelable, std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<bool> absolute);
#endif

#if ENABLE(DEVICE_ORIENTATION)
    using PermissionState = DeviceOrientationOrMotionPermissionState;
    using PermissionPromise = DOMPromiseDeferred<IDLEnumeration<PermissionState>>;
    static void requestPermission(Document&, PermissionPromise&&);
#endif

private:
    DeviceOrientationEvent();
    DeviceOrientationEvent(const AtomString& eventType, DeviceOrientationData*);

    RefPtr<DeviceOrientationData> m_orientation;
};

} // namespace WebCore
