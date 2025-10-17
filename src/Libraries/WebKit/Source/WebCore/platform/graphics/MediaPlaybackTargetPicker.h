/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#ifndef MediaPlaybackTargetPicker_h
#define MediaPlaybackTargetPicker_h

#if ENABLE(WIRELESS_PLAYBACK_TARGET)

#include "PlatformView.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Ref.h>
#include <wtf/RunLoop.h>

namespace WebCore {

class FloatRect;
class MediaPlaybackTarget;

class MediaPlaybackTargetPicker : public CanMakeCheckedPtr<MediaPlaybackTargetPicker> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaPlaybackTargetPicker);
public:
    class Client {
    protected:
        virtual ~Client() = default;

    public:
        virtual void setPlaybackTarget(Ref<MediaPlaybackTarget>&&) = 0;
        virtual void externalOutputDeviceAvailableDidChange(bool) = 0;
        virtual void playbackTargetPickerWasDismissed() = 0;
    };

    virtual ~MediaPlaybackTargetPicker();

    virtual void showPlaybackTargetPicker(PlatformView*, const FloatRect&, bool checkActiveRoute, bool useDarkAppearance);
    virtual void startingMonitoringPlaybackTargets();
    virtual void stopMonitoringPlaybackTargets();
    virtual void invalidatePlaybackTargets();

    void availableDevicesDidChange() { addPendingAction(OutputDeviceAvailabilityChanged); }
    void currentDeviceDidChange() { addPendingAction(CurrentDeviceDidChange); }
    void playbackTargetPickerWasDismissed() { addPendingAction(PlaybackTargetPickerWasDismissed); }

protected:
    explicit MediaPlaybackTargetPicker(Client&);

    enum ActionType {
        OutputDeviceAvailabilityChanged = 1 << 0,
        CurrentDeviceDidChange = 1 << 1,
        PlaybackTargetPickerWasDismissed = 1 << 2,
    };
    typedef unsigned PendingActionFlags;

    void addPendingAction(PendingActionFlags);
    void pendingActionTimerFired();
    Client* client() const { return m_client; }
    void setClient(Client* client) { m_client = client; }

private:
    virtual bool externalOutputDeviceAvailable() = 0;
    virtual Ref<MediaPlaybackTarget> playbackTarget() = 0;

    PendingActionFlags m_pendingActionFlags { 0 };
    Client* m_client;
    RunLoop::Timer m_pendingActionTimer;
};

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)

#endif // MediaPlaybackTargetPicker_h
