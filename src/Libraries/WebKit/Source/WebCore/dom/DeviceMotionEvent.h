/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#include "Document.h"
#include "Event.h"
#include "IDLTypes.h"

namespace WebCore {

class DeviceMotionData;
template<typename IDLType> class DOMPromiseDeferred;

class DeviceMotionEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DeviceMotionEvent);
public:
    virtual ~DeviceMotionEvent();

    // FIXME: Merge this with DeviceMotionData::Acceleration
    struct Acceleration {
        std::optional<double> x;
        std::optional<double> y;
        std::optional<double> z;
    };

    // FIXME: Merge this with DeviceMotionData::RotationRate
    struct RotationRate {
        std::optional<double> alpha;
        std::optional<double> beta;
        std::optional<double> gamma;
    };

    static Ref<DeviceMotionEvent> create(const AtomString& eventType, DeviceMotionData* deviceMotionData)
    {
        return adoptRef(*new DeviceMotionEvent(eventType, deviceMotionData));
    }

    static Ref<DeviceMotionEvent> createForBindings()
    {
        return adoptRef(*new DeviceMotionEvent);
    }

    std::optional<Acceleration> acceleration() const;
    std::optional<Acceleration> accelerationIncludingGravity() const;
    std::optional<RotationRate> rotationRate() const;
    std::optional<double> interval() const;

    void initDeviceMotionEvent(const AtomString& type, bool bubbles, bool cancelable, std::optional<Acceleration>&&, std::optional<Acceleration>&&, std::optional<RotationRate>&&, std::optional<double>);

#if ENABLE(DEVICE_ORIENTATION)
    using PermissionState = DeviceOrientationOrMotionPermissionState;
    using PermissionPromise = DOMPromiseDeferred<IDLEnumeration<PermissionState>>;
    static void requestPermission(Document&, PermissionPromise&&);
#endif

private:
    DeviceMotionEvent();
    DeviceMotionEvent(const AtomString& eventType, DeviceMotionData*);

    RefPtr<DeviceMotionData> m_deviceMotionData;
};

} // namespace WebCore
