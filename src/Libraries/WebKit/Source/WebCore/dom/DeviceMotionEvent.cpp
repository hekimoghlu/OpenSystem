/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#include "DeviceMotionEvent.h"

#include "DeviceMotionData.h"
#include "DeviceOrientationAndMotionAccessController.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalDOMWindow.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DeviceMotionEvent);

DeviceMotionEvent::~DeviceMotionEvent() = default;

DeviceMotionEvent::DeviceMotionEvent()
#if ENABLE(DEVICE_ORIENTATION)
    : Event(EventInterfaceType::DeviceMotionEvent)
#else
    // FIXME: ENABLE(DEVICE_ORIENTATION) seems to be in a strange state where
    // it is half-guarded by #ifdefs. DeviceMotionEvent.idl is guarded
    // but DeviceMotionEvent.cpp itself is required by unguarded code.
    : Event(EventInterfaceType::Event)
#endif
    , m_deviceMotionData(DeviceMotionData::create())
{
}

DeviceMotionEvent::DeviceMotionEvent(const AtomString& eventType, DeviceMotionData* deviceMotionData)
#if ENABLE(DEVICE_ORIENTATION)
    : Event(EventInterfaceType::DeviceMotionEvent, eventType, CanBubble::No, IsCancelable::No)
#else
    // FIXME: ENABLE(DEVICE_ORIENTATION) seems to be in a strange state where
    // it is half-guarded by #ifdefs. DeviceMotionEvent.idl is guarded
    // but DeviceMotionEvent.cpp itself is required by unguarded code.
    : Event(EventInterfaceType::Event, eventType, CanBubble::No, IsCancelable::No)
#endif
    , m_deviceMotionData(deviceMotionData)
{
}

static std::optional<DeviceMotionEvent::Acceleration> convert(const DeviceMotionData::Acceleration* acceleration)
{
    if (!acceleration)
        return std::nullopt;

    return DeviceMotionEvent::Acceleration { acceleration->x(), acceleration->y(), acceleration->z() };
}

static std::optional<DeviceMotionEvent::RotationRate> convert(const DeviceMotionData::RotationRate* rotationRate)
{
    if (!rotationRate)
        return std::nullopt;

    return DeviceMotionEvent::RotationRate { rotationRate->alpha(), rotationRate->beta(), rotationRate->gamma() };
}

static RefPtr<DeviceMotionData::Acceleration> convert(std::optional<DeviceMotionEvent::Acceleration>&& acceleration)
{
    if (!acceleration)
        return nullptr;

    if (!acceleration->x && !acceleration->y && !acceleration->z)
        return nullptr;

    return DeviceMotionData::Acceleration::create(acceleration->x, acceleration->y, acceleration->z);
}

static RefPtr<DeviceMotionData::RotationRate> convert(std::optional<DeviceMotionEvent::RotationRate>&& rotationRate)
{
    if (!rotationRate)
        return nullptr;

    if (!rotationRate->alpha && !rotationRate->beta && !rotationRate->gamma)
        return nullptr;

    return DeviceMotionData::RotationRate::create(rotationRate->alpha, rotationRate->beta, rotationRate->gamma);
}

std::optional<DeviceMotionEvent::Acceleration> DeviceMotionEvent::acceleration() const
{
    RefPtr acceleration = m_deviceMotionData->acceleration();
    return convert(acceleration.get());
}

std::optional<DeviceMotionEvent::Acceleration> DeviceMotionEvent::accelerationIncludingGravity() const
{
    RefPtr accelerationIncludingGravity = m_deviceMotionData->accelerationIncludingGravity();
    return convert(accelerationIncludingGravity.get());
}

std::optional<DeviceMotionEvent::RotationRate> DeviceMotionEvent::rotationRate() const
{
    RefPtr rotationRate = m_deviceMotionData->rotationRate();
    return convert(rotationRate.get());
}

std::optional<double> DeviceMotionEvent::interval() const
{
    return m_deviceMotionData->interval();
}

void DeviceMotionEvent::initDeviceMotionEvent(const AtomString& type, bool bubbles, bool cancelable, std::optional<DeviceMotionEvent::Acceleration>&& acceleration, std::optional<DeviceMotionEvent::Acceleration>&& accelerationIncludingGravity, std::optional<DeviceMotionEvent::RotationRate>&& rotationRate, std::optional<double> interval)
{
    if (isBeingDispatched())
        return;

    initEvent(type, bubbles, cancelable);
    m_deviceMotionData = DeviceMotionData::create(convert(WTFMove(acceleration)), convert(WTFMove(accelerationIncludingGravity)), convert(WTFMove(rotationRate)), interval);
}

#if ENABLE(DEVICE_ORIENTATION)
void DeviceMotionEvent::requestPermission(Document& document, PermissionPromise&& promise)
{
    RefPtr window = document.domWindow();
    if (!window || !document.page())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "No browsing context"_s });

    String errorMessage;
    if (!window->isAllowedToUseDeviceMotion(errorMessage)) {
        document.addConsoleMessage(MessageSource::JS, MessageLevel::Warning, makeString("Call to requestPermission() failed, reason: "_s, errorMessage, '.'));
        return promise.resolve(PermissionState::Denied);
    }

    document.deviceOrientationAndMotionAccessController().shouldAllowAccess(document, [promise = WTFMove(promise)](auto permissionState) mutable {
        if (permissionState == PermissionState::Prompt)
            return promise.reject(Exception { ExceptionCode::NotAllowedError, "Requesting device motion access requires a user gesture to prompt"_s });
        promise.resolve(permissionState);
    });
}
#endif

} // namespace WebCore
