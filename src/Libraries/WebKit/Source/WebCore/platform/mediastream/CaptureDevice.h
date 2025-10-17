/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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

#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CaptureDevice {
public:
    enum class DeviceType : uint8_t { Unknown, Microphone, Speaker, Camera, Screen, Window, SystemAudio };
    static bool isScreenShareType(DeviceType type) { return type == DeviceType::Screen || type == DeviceType::Window || type == DeviceType::SystemAudio; }

    CaptureDevice(const String& persistentId, DeviceType type, const String& label, const String& groupId = emptyString(), bool isEnabled = false, bool isDefault = false, bool isMock = false, bool isEphemeral = false)
        : m_persistentId(persistentId)
        , m_type(type)
        , m_label(label)
        , m_groupId(groupId)
        , m_enabled(isEnabled)
        , m_default(isDefault)
        , m_isMockDevice(isMock)
        , m_isEphemeral(isEphemeral || isScreenShareType(m_type))
    {
    }

    CaptureDevice() = default;

    void setPersistentId(const String& persistentId) { m_persistentId = persistentId; }
    const String& persistentId() const { return m_persistentId; }

    void setLabel(const String& label) { m_label = label; }
    const String& label() const
    {
        static NeverDestroyed<String> airPods(MAKE_STATIC_STRING_IMPL("AirPods"));

        if ((m_type == DeviceType::Microphone || m_type == DeviceType::Speaker) && m_label.contains(airPods.get()))
            return airPods;

        return m_label;
    }

    void setGroupId(const String& groupId) { m_groupId = groupId; }
    const String& groupId() const { return m_groupId.isEmpty() ? m_persistentId : m_groupId; }

    DeviceType type() const { return m_type; }

    bool enabled() const { return m_enabled; }
    void setEnabled(bool enabled) { m_enabled = enabled; }

    bool isDefault() const { return m_default; }
    void setIsDefault(bool isDefault) { m_default = isDefault; }

    bool isMockDevice() const { return m_isMockDevice; }
    void setIsMockDevice(bool isMockDevice) { m_isMockDevice = isMockDevice; }

    bool isEphemeral() const { return m_isEphemeral; }
    void setIsEphemeral(bool isEphemeral) { m_isEphemeral = isEphemeral; }

    bool isInputDevice() const
    {
        return m_type == DeviceType::Microphone || m_type == DeviceType::Camera;
    }

    explicit operator bool() const { return m_type != DeviceType::Unknown; }

    CaptureDevice isolatedCopy() &&;

protected:
    String m_persistentId;
    DeviceType m_type { DeviceType::Unknown };
    String m_label;
    String m_groupId;
    bool m_enabled { false };
    bool m_default { false };
    bool m_isMockDevice { false };
    bool m_isEphemeral { false };
};

inline bool haveDevicesChanged(const Vector<CaptureDevice>& oldDevices, const Vector<CaptureDevice>& newDevices)
{
    if (oldDevices.size() != newDevices.size())
        return true;

    for (auto& newDevice : newDevices) {
        if (newDevice.type() != CaptureDevice::DeviceType::Camera && newDevice.type() != CaptureDevice::DeviceType::Microphone)
            continue;

        auto index = oldDevices.findIf([&newDevice](auto& oldDevice) {
            return newDevice.persistentId() == oldDevice.persistentId() && newDevice.enabled() != oldDevice.enabled();
        });

        if (index == notFound)
            return true;
    }

    return false;
}

inline CaptureDevice CaptureDevice::isolatedCopy() &&
{
    return {
        WTFMove(m_persistentId).isolatedCopy(),
        m_type,
        WTFMove(m_label).isolatedCopy(),
        WTFMove(m_groupId).isolatedCopy(),
        m_enabled,
        m_default,
        m_isMockDevice,
        m_isEphemeral
    };
}

} // namespace WebCore
