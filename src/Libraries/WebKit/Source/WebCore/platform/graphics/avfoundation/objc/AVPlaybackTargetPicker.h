/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS AVOutputContext;
OBJC_CLASS NSView;

namespace WebCore {
class AVPlaybackTargetPicker;
class AVPlaybackTargetPickerClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AVPlaybackTargetPicker> : std::true_type { };
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AVPlaybackTargetPickerClient> : std::true_type { };
}

namespace WebCore {

class FloatRect;

class AVPlaybackTargetPickerClient : public CanMakeWeakPtr<AVPlaybackTargetPickerClient> {
protected:
    virtual ~AVPlaybackTargetPickerClient() = default;

public:
    virtual void pickerWasDismissed() = 0;
    virtual void availableDevicesChanged() = 0;
    virtual void currentDeviceChanged() = 0;
};

class AVPlaybackTargetPicker : public CanMakeWeakPtr<AVPlaybackTargetPicker> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(AVPlaybackTargetPicker);
    WTF_MAKE_NONCOPYABLE(AVPlaybackTargetPicker);
public:
    explicit AVPlaybackTargetPicker(AVPlaybackTargetPickerClient& client)
        : m_client(client)
    {
    }
    virtual ~AVPlaybackTargetPicker() = default;

    virtual void showPlaybackTargetPicker(NSView *, const FloatRect&, bool checkActiveRoute, bool useDarkAppearancebool) = 0;
    virtual void startingMonitoringPlaybackTargets() = 0;
    virtual void stopMonitoringPlaybackTargets() = 0;
    virtual void invalidatePlaybackTargets() = 0;
    virtual bool externalOutputDeviceAvailable() = 0;

    virtual AVOutputContext* outputContext() = 0;

    WeakPtr<AVPlaybackTargetPickerClient> client() const { return m_client; }

private:
    WeakPtr<AVPlaybackTargetPickerClient> m_client;
};

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
