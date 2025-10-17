/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#include "MediaPlaybackTargetPickerMock.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#include "FloatRect.h"
#include "Logging.h"
#include "MediaPlaybackTargetMock.h"
#include "WebMediaSessionManager.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaPlaybackTargetPickerMock);

static const Seconds timerInterval { 100_ms };

MediaPlaybackTargetPickerMock::MediaPlaybackTargetPickerMock(MediaPlaybackTargetPicker::Client& client)
    : MediaPlaybackTargetPicker(client)
{
    LOG(Media, "MediaPlaybackTargetPickerMock::MediaPlaybackTargetPickerMock");
}

MediaPlaybackTargetPickerMock::~MediaPlaybackTargetPickerMock()
{
    LOG(Media, "MediaPlaybackTargetPickerMock::~MediaPlaybackTargetPickerMock");
    setClient(nullptr);
}

bool MediaPlaybackTargetPickerMock::externalOutputDeviceAvailable()
{
    LOG(Media, "MediaPlaybackTargetPickerMock::externalOutputDeviceAvailable");
    return m_state == MediaPlaybackTargetContext::MockState::OutputDeviceAvailable;
}

Ref<MediaPlaybackTarget> MediaPlaybackTargetPickerMock::playbackTarget()
{
    LOG(Media, "MediaPlaybackTargetPickerMock::playbackTarget");
    return WebCore::MediaPlaybackTargetMock::create(MediaPlaybackTargetContextMock { m_deviceName, m_state });
}

void MediaPlaybackTargetPickerMock::showPlaybackTargetPicker(PlatformView*, const FloatRect&, bool checkActiveRoute, bool useDarkAppearance)
{
    if (!client() || m_showingMenu)
        return;

#if LOG_DISABLED
    UNUSED_PARAM(checkActiveRoute);
    UNUSED_PARAM(useDarkAppearance);
#endif

    LOG(Media, "MediaPlaybackTargetPickerMock::showPlaybackTargetPicker - checkActiveRoute = %i, useDarkAppearance = %i", (int)checkActiveRoute, (int)useDarkAppearance);

    m_showingMenu = true;
    callOnMainThread([this, weakThis = WeakPtr { *this }] {
        if (!weakThis)
            return;

        m_showingMenu = false;
        currentDeviceDidChange();
    });
}

void MediaPlaybackTargetPickerMock::startingMonitoringPlaybackTargets()
{
    LOG(Media, "MediaPlaybackTargetPickerMock::startingMonitoringPlaybackTargets");

    callOnMainThread([this, weakThis = WeakPtr { *this }] {
        if (!weakThis)
            return;

        if (m_state == MediaPlaybackTargetContext::MockState::OutputDeviceAvailable)
            availableDevicesDidChange();

        if (!m_deviceName.isEmpty() && m_state != MediaPlaybackTargetContext::MockState::Unknown)
            currentDeviceDidChange();
    });
}

void MediaPlaybackTargetPickerMock::stopMonitoringPlaybackTargets()
{
    LOG(Media, "MediaPlaybackTargetPickerMock::stopMonitoringPlaybackTargets");
}

void MediaPlaybackTargetPickerMock::invalidatePlaybackTargets()
{
    LOG(Media, "MediaPlaybackTargetPickerMock::invalidatePlaybackTargets");
    setState(emptyString(), MediaPlaybackTargetContext::MockState::Unknown);
}

void MediaPlaybackTargetPickerMock::setState(const String& deviceName, MediaPlaybackTargetContext::MockState state)
{
    LOG(Media, "MediaPlaybackTargetPickerMock::setState - name = %s, state = 0x%x", deviceName.utf8().data(), (unsigned)state);

    callOnMainThread([this, weakThis = WeakPtr { *this }, state, deviceName] {
        if (!weakThis)
            return;

        if (deviceName != m_deviceName && state != MediaPlaybackTargetContext::MockState::Unknown) {
            m_deviceName = deviceName;
            currentDeviceDidChange();
        }

        if (m_state != state) {
            m_state = state;
            availableDevicesDidChange();
        }
    });
}

void MediaPlaybackTargetPickerMock::dismissPopup()
{
    if (!m_showingMenu)
        return;

    m_showingMenu = false;
    currentDeviceDidChange();
}

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)
