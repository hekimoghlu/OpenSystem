/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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

#if USE(GLIB) && ENABLE(MEDIA_SESSION)

#include "MediaSessionIdentifier.h"
#include "PlatformMediaSession.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>

namespace WebCore {

enum class MediaSessionGLibMprisRegistrationEligiblilty : uint8_t {
    Eligible,
    NotEligible,
};

class MediaSessionManagerGLib;

class MediaSessionGLib {
    WTF_MAKE_TZONE_ALLOCATED(MediaSessionGLib);

public:
    static std::unique_ptr<MediaSessionGLib> create(MediaSessionManagerGLib&, MediaSessionIdentifier);

    explicit MediaSessionGLib(MediaSessionManagerGLib&, GRefPtr<GDBusConnection>&&, MediaSessionIdentifier);
    ~MediaSessionGLib();

    MediaSessionManagerGLib& manager() const { return m_manager; }

    GVariant* getPlaybackStatusAsGVariant(std::optional<const PlatformMediaSession*>);
    GVariant* getMetadataAsGVariant(std::optional<NowPlayingInfo>);
    GVariant* getPositionAsGVariant();
    GVariant* canSeekAsGVariant();

    void emitPositionChanged(double time);
    void updateNowPlaying(NowPlayingInfo&);
    void playbackStatusChanged(PlatformMediaSession&);

    void unregisterMprisSession();

    using MprisRegistrationEligiblilty = MediaSessionGLibMprisRegistrationEligiblilty;
    void setMprisRegistrationEligibility(MprisRegistrationEligiblilty eligibility) { m_registrationEligibility = eligibility; }
    MprisRegistrationEligiblilty mprisRegistrationEligibility() const { return m_registrationEligibility; }
private:
    void emitPropertiesChanged(GRefPtr<GVariant>&&);
    std::optional<NowPlayingInfo> nowPlayingInfo();
    bool ensureMprisSessionRegistered();

    MediaSessionIdentifier m_identifier;
    MediaSessionManagerGLib& m_manager;
    GRefPtr<GDBusConnection> m_connection;
    MprisRegistrationEligiblilty m_registrationEligibility { MprisRegistrationEligiblilty::NotEligible };
    String m_instanceId;
    unsigned m_ownerId { 0 };
    unsigned m_rootRegistrationId { 0 };
    unsigned m_playerRegistrationId { 0 };
};

String convertEnumerationToString(MediaSessionGLib::MprisRegistrationEligiblilty);

} // namespace WebCore

namespace WTF {

template<typename Type>
struct LogArgument;

template <>
struct LogArgument<WebCore::MediaSessionGLib::MprisRegistrationEligiblilty> {
    static String toString(const WebCore::MediaSessionGLib::MprisRegistrationEligiblilty state)
    {
        return convertEnumerationToString(state);
    }
};

} // namespace WTF

#endif // USE(GLIB) && ENABLE(MEDIA_SESSION)
