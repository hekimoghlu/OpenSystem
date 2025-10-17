/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#include "DeviceOrientationEvent.h"

#include "DeviceOrientationAndMotionAccessController.h"
#include "DeviceOrientationData.h"
#include "Document.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalDOMWindow.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DeviceOrientationEvent);

DeviceOrientationEvent::~DeviceOrientationEvent() = default;

DeviceOrientationEvent::DeviceOrientationEvent()
#if ENABLE(DEVICE_ORIENTATION)
    : Event(EventInterfaceType::DeviceOrientationEvent)
#else
    // FIXME: ENABLE(DEVICE_ORIENTATION) seems to be in a strange state where
    // it is half-guarded by #ifdefs. DeviceOrientationEvent.idl is guarded
    // but DeviceOrientationEvent.cpp itself is required by unguarded code.
    : Event(EventInterfaceType::Event)
#endif
    , m_orientation(DeviceOrientationData::create())
{
}

DeviceOrientationEvent::DeviceOrientationEvent(const AtomString& eventType, DeviceOrientationData* orientation)
#if ENABLE(DEVICE_ORIENTATION)
    : Event(EventInterfaceType::DeviceOrientationEvent, eventType, CanBubble::No, IsCancelable::No)
#else
    // FIXME: ENABLE(DEVICE_ORIENTATION) seems to be in a strange state where
    // it is half-guarded by #ifdefs. DeviceOrientationEvent.idl is guarded
    // but DeviceOrientationEvent.cpp itself is required by unguarded code.
    : Event(EventInterfaceType::Event, eventType, CanBubble::No, IsCancelable::No)
#endif
    , m_orientation(orientation)
{
}

std::optional<double> DeviceOrientationEvent::alpha() const
{
    return m_orientation->alpha();
}

std::optional<double> DeviceOrientationEvent::beta() const
{
    return m_orientation->beta();
}

std::optional<double> DeviceOrientationEvent::gamma() const
{
    return m_orientation->gamma();
}

#if PLATFORM(IOS_FAMILY)

std::optional<double> DeviceOrientationEvent::compassHeading() const
{
    return m_orientation->compassHeading();
}

std::optional<double> DeviceOrientationEvent::compassAccuracy() const
{
    return m_orientation->compassAccuracy();
}

void DeviceOrientationEvent::initDeviceOrientationEvent(const AtomString& type, bool bubbles, bool cancelable, std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<double> compassHeading, std::optional<double> compassAccuracy)
{
    if (isBeingDispatched())
        return;

    initEvent(type, bubbles, cancelable);
    m_orientation = DeviceOrientationData::create(alpha, beta, gamma, compassHeading, compassAccuracy);
}

#else

std::optional<bool> DeviceOrientationEvent::absolute() const
{
    return m_orientation->absolute();
}

void DeviceOrientationEvent::initDeviceOrientationEvent(const AtomString& type, bool bubbles, bool cancelable, std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<bool> absolute)
{
    if (isBeingDispatched())
        return;

    initEvent(type, bubbles, cancelable);
    m_orientation = DeviceOrientationData::create(alpha, beta, gamma, absolute);
}

#endif

#if ENABLE(DEVICE_ORIENTATION)
void DeviceOrientationEvent::requestPermission(Document& document, PermissionPromise&& promise)
{
    RefPtr window = document.domWindow();
    if (!window || !document.page())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "No browsing context"_s });

    String errorMessage;
    if (!window->isAllowedToUseDeviceOrientation(errorMessage)) {
        document.addConsoleMessage(MessageSource::JS, MessageLevel::Warning, makeString("Call to requestPermission() failed, reason: "_s, errorMessage, '.'));
        return promise.resolve(PermissionState::Denied);
    }

    document.deviceOrientationAndMotionAccessController().shouldAllowAccess(document, [promise = WTFMove(promise)](PermissionState permissionState) mutable {
        if (permissionState == PermissionState::Prompt)
            return promise.reject(Exception { ExceptionCode::NotAllowedError, "Requesting device orientation access requires a user gesture to prompt"_s });
        promise.resolve(permissionState);
    });
}
#endif

} // namespace WebCore
