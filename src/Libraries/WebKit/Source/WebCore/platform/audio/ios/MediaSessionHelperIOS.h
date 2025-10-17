/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

#if PLATFORM(IOS_FAMILY)

#include <wtf/WeakHashSet.h>

namespace WebCore {
class MediaSessionHelperClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MediaSessionHelperClient> : std::true_type { };
}

namespace WebCore {

class MediaPlaybackTarget;

class MediaSessionHelperClient : public CanMakeWeakPtr<MediaSessionHelperClient> {
public:
    virtual ~MediaSessionHelperClient() = default;

    enum class SuspendedUnderLock : bool { No, Yes };
    virtual void applicationWillEnterForeground(SuspendedUnderLock) = 0;
    virtual void applicationDidEnterBackground(SuspendedUnderLock) = 0;
    virtual void applicationWillBecomeInactive() = 0;
    virtual void applicationDidBecomeActive() = 0;

    enum class HasAvailableTargets : bool { No, Yes };
    virtual void externalOutputDeviceAvailableDidChange(HasAvailableTargets) = 0;

    enum class PlayingToAutomotiveHeadUnit : bool { No, Yes };
    virtual void isPlayingToAutomotiveHeadUnitDidChange(PlayingToAutomotiveHeadUnit) = 0;

    enum class ShouldPause : bool { No, Yes };
    virtual void activeAudioRouteDidChange(ShouldPause) = 0;

    enum class SupportsAirPlayVideo : bool { No, Yes };
    virtual void activeVideoRouteDidChange(SupportsAirPlayVideo, Ref<MediaPlaybackTarget>&&) = 0;

    enum class SupportsSpatialAudioPlayback : bool { No, Yes };
    virtual void activeAudioRouteSupportsSpatialPlaybackDidChange(SupportsSpatialAudioPlayback) = 0;
};

class WEBCORE_EXPORT MediaSessionHelper : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<MediaSessionHelper> {
public:
    static MediaSessionHelper& sharedHelper();
    static void setSharedHelper(Ref<MediaSessionHelper>&&);
    static void resetSharedHelper();

    MediaSessionHelper() = default;
    explicit MediaSessionHelper(bool isExternalOutputDeviceAvailable);
    virtual ~MediaSessionHelper() = default;

    void addClient(MediaSessionHelperClient&);
    void removeClient(MediaSessionHelperClient&);

    void startMonitoringWirelessRoutes();
    void stopMonitoringWirelessRoutes();

    enum class ShouldOverride : bool { No, Yes };
    void providePresentingApplicationPID(int pid) { providePresentingApplicationPID(pid, ShouldOverride::No); }
    virtual void providePresentingApplicationPID(int, ShouldOverride) = 0;

    void setIsExternalOutputDeviceAvailable(bool);

    bool isMonitoringWirelessRoutes() const { return m_monitoringWirelessRoutesCount; }
    bool isExternalOutputDeviceAvailable() const { return m_isExternalOutputDeviceAvailable; }
    bool activeVideoRouteSupportsAirPlayVideo() const { return m_activeVideoRouteSupportsAirPlayVideo; }
    bool isPlayingToAutomotiveHeadUnit() const { return m_isPlayingToAutomotiveHeadUnit; }

    MediaPlaybackTarget* playbackTarget() const { return m_playbackTarget.get(); }

    using HasAvailableTargets = MediaSessionHelperClient::HasAvailableTargets;
    using PlayingToAutomotiveHeadUnit = MediaSessionHelperClient::PlayingToAutomotiveHeadUnit;
    using ShouldPause = MediaSessionHelperClient::ShouldPause;
    using SupportsAirPlayVideo = MediaSessionHelperClient::SupportsAirPlayVideo;
    using SuspendedUnderLock = MediaSessionHelperClient::SuspendedUnderLock;
    using SupportsSpatialAudioPlayback = MediaSessionHelperClient::SupportsSpatialAudioPlayback;

    void activeAudioRouteDidChange(ShouldPause);
    void applicationWillEnterForeground(SuspendedUnderLock);
    void applicationDidEnterBackground(SuspendedUnderLock);
    void applicationWillBecomeInactive();
    void applicationDidBecomeActive();

    void setActiveAudioRouteSupportsSpatialPlayback(bool);
    void updateActiveAudioRouteSupportsSpatialPlayback();

protected:
    void externalOutputDeviceAvailableDidChange(HasAvailableTargets);
    void isPlayingToAutomotiveHeadUnitDidChange(PlayingToAutomotiveHeadUnit);
    void activeVideoRouteDidChange(SupportsAirPlayVideo, Ref<MediaPlaybackTarget>&&);
    void activeAudioRouteSupportsSpatialPlaybackDidChange(SupportsSpatialAudioPlayback);

private:
    virtual void startMonitoringWirelessRoutesInternal() = 0;
    virtual void stopMonitoringWirelessRoutesInternal() = 0;

    WeakHashSet<MediaSessionHelperClient> m_clients;
    bool m_isExternalOutputDeviceAvailable { false };
    uint32_t m_monitoringWirelessRoutesCount { 0 };
    bool m_activeVideoRouteSupportsAirPlayVideo { false };
    bool m_isPlayingToAutomotiveHeadUnit { false };
    SupportsSpatialAudioPlayback m_activeAudioRouteSupportsSpatialPlayback { SupportsSpatialAudioPlayback::No };
    RefPtr<MediaPlaybackTarget> m_playbackTarget;
};

inline MediaSessionHelper::MediaSessionHelper(bool isExternalOutputDeviceAvailable)
    : m_isExternalOutputDeviceAvailable(isExternalOutputDeviceAvailable)
{
}

inline void MediaSessionHelper::setIsExternalOutputDeviceAvailable(bool isExternalOutputDeviceAvailable)
{
    m_isExternalOutputDeviceAvailable = isExternalOutputDeviceAvailable;
}

}

#endif
