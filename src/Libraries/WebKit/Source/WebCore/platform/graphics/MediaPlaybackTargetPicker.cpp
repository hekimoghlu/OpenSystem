/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#include "MediaPlaybackTargetPicker.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET)

#include "Logging.h"
#include "MediaPlaybackTarget.h"

namespace WebCore {

static const Seconds pendingActionInterval { 100_ms };

MediaPlaybackTargetPicker::MediaPlaybackTargetPicker(Client& client)
    : m_client(&client)
    , m_pendingActionTimer(RunLoop::main(), this, &MediaPlaybackTargetPicker::pendingActionTimerFired)
{
}

MediaPlaybackTargetPicker::~MediaPlaybackTargetPicker()
{
    m_pendingActionTimer.stop();
    m_client = nullptr;
}

void MediaPlaybackTargetPicker::pendingActionTimerFired()
{
    LOG(Media, "MediaPlaybackTargetPicker::pendingActionTimerFired - flags = 0x%x", m_pendingActionFlags);

    PendingActionFlags pendingActions = m_pendingActionFlags;
    m_pendingActionFlags = 0;

    if (pendingActions & CurrentDeviceDidChange)
        m_client->setPlaybackTarget(playbackTarget());

    if (pendingActions & OutputDeviceAvailabilityChanged)
        m_client->externalOutputDeviceAvailableDidChange(externalOutputDeviceAvailable());

    if (pendingActions & PlaybackTargetPickerWasDismissed)
        m_client->playbackTargetPickerWasDismissed();
}

void MediaPlaybackTargetPicker::addPendingAction(PendingActionFlags action)
{
    if (!m_client)
        return;

    m_pendingActionFlags |= action;
    m_pendingActionTimer.startOneShot(pendingActionInterval);
}

void MediaPlaybackTargetPicker::showPlaybackTargetPicker(PlatformView*, const FloatRect&, bool, bool)
{
    ASSERT_NOT_REACHED();
}

void MediaPlaybackTargetPicker::startingMonitoringPlaybackTargets()
{
    ASSERT_NOT_REACHED();
}

void MediaPlaybackTargetPicker::stopMonitoringPlaybackTargets()
{
    ASSERT_NOT_REACHED();
}

void MediaPlaybackTargetPicker::invalidatePlaybackTargets()
{
    ASSERT_NOT_REACHED();
}

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
