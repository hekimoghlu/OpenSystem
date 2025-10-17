/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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

#include "MediaPlaybackTargetContext.h"
#include "MediaPlaybackTargetPicker.h"
#include "MediaPlaybackTargetPickerMock.h"
#include "MediaProducer.h"
#include "PlaybackTargetClientContextIdentifier.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/RunLoop.h>

namespace WebCore {

struct ClientState;
class IntRect;
class WebMediaSessionLogger;
class WebMediaSessionManagerClient;

class WebMediaSessionManager : public MediaPlaybackTargetPicker::Client, public CanMakeCheckedPtr<WebMediaSessionManager> {
    WTF_MAKE_NONCOPYABLE(WebMediaSessionManager);
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebMediaSessionManager);
public:

    WEBCORE_EXPORT static WebMediaSessionManager& shared();

    WEBCORE_EXPORT void setMockMediaPlaybackTargetPickerEnabled(bool);
    WEBCORE_EXPORT void setMockMediaPlaybackTargetPickerState(const String&, MediaPlaybackTargetContext::MockState);
    WEBCORE_EXPORT void mockMediaPlaybackTargetPickerDismissPopup();

    WEBCORE_EXPORT std::optional<PlaybackTargetClientContextIdentifier> addPlaybackTargetPickerClient(WebMediaSessionManagerClient&, PlaybackTargetClientContextIdentifier);
    WEBCORE_EXPORT void removePlaybackTargetPickerClient(WebMediaSessionManagerClient&, PlaybackTargetClientContextIdentifier);
    WEBCORE_EXPORT void removeAllPlaybackTargetPickerClients(WebMediaSessionManagerClient&);
    WEBCORE_EXPORT void showPlaybackTargetPicker(WebMediaSessionManagerClient&, PlaybackTargetClientContextIdentifier, const IntRect&, bool, bool);
    WEBCORE_EXPORT void clientStateDidChange(WebMediaSessionManagerClient&, PlaybackTargetClientContextIdentifier, WebCore::MediaProducerMediaStateFlags);

    bool alwaysOnLoggingAllowed() const;

protected:
    WebMediaSessionManager();
    virtual ~WebMediaSessionManager();

    virtual WebCore::MediaPlaybackTargetPicker& platformPicker() = 0;
    static WebMediaSessionManager& platformManager();

private:

    WebCore::MediaPlaybackTargetPicker& targetPicker();
    WebCore::MediaPlaybackTargetPickerMock& mockPicker();

    // MediaPlaybackTargetPicker::Client
    void setPlaybackTarget(Ref<WebCore::MediaPlaybackTarget>&&) final;
    void externalOutputDeviceAvailableDidChange(bool) final;
    void playbackTargetPickerWasDismissed() final;

    size_t find(WebMediaSessionManagerClient*, PlaybackTargetClientContextIdentifier);

    void configurePlaybackTargetClients();
    void configureNewClients();
    void configurePlaybackTargetMonitoring();
    void configureWatchdogTimer();

    enum ConfigurationTaskFlags {
        NoTask = 0,
        InitialConfigurationTask = 1 << 0,
        TargetClientsConfigurationTask = 1 << 1,
        TargetMonitoringConfigurationTask = 1 << 2,
        WatchdogTimerConfigurationTask = 1 << 3,
    };
    typedef unsigned ConfigurationTasks;
    String toString(ConfigurationTasks);

    void scheduleDelayedTask(ConfigurationTasks);
    void taskTimerFired();

    void watchdogTimerFired();

    WebMediaSessionLogger& logger();

    RunLoop::Timer m_taskTimer;
    RunLoop::Timer m_watchdogTimer;

    Vector<std::unique_ptr<ClientState>> m_clientState;
    RefPtr<MediaPlaybackTarget> m_playbackTarget;
    std::unique_ptr<WebCore::MediaPlaybackTargetPickerMock> m_pickerOverride;
    ConfigurationTasks m_taskFlags { NoTask };
    std::unique_ptr<WebMediaSessionLogger> m_logger;
    Seconds m_currentWatchdogInterval;
    bool m_externalOutputDeviceAvailable { false };
    bool m_targetChanged { false };
    bool m_playbackTargetPickerDismissed { false };
    bool m_mockPickerEnabled { false };
};

String mediaProducerStateString(WebCore::MediaProducerMediaStateFlags);

} // namespace WebCore

namespace WTF {

template<typename> struct LogArgument;

template<> struct LogArgument<WebCore::MediaProducerMediaStateFlags> {
    static String toString(WebCore::MediaProducerMediaStateFlags flags) { return WebCore::mediaProducerStateString(flags); }
};

} // namespace WTF

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)
