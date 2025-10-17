/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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

#if USE(GSTREAMER_WEBRTC)

#include "GRefPtrGStreamer.h"
#include "GStreamerRTPPacketizer.h"
#include "GStreamerWebRTCUtils.h"
#include "MediaStreamTrackPrivate.h"
#include "RTCRtpCapabilities.h"

#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class MediaStreamTrack;

class RealtimeOutgoingMediaSourceGStreamer : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RealtimeOutgoingMediaSourceGStreamer, WTF::DestructionThread::Main>, public MediaStreamTrackPrivateObserver {
public:
    ~RealtimeOutgoingMediaSourceGStreamer();
    void start();
    void stop();

    const RefPtr<MediaStreamTrackPrivate>& track() const { return m_track; }

    const String& mediaStreamID() const { return m_mediaStreamId; }
    const GRefPtr<GstCaps>& allowedCaps() const;
    WARN_UNUSED_RETURN GRefPtr<GstCaps> rtpCaps() const;

    void link();
    const GRefPtr<GstPad>& pad() const { return m_webrtcSinkPad; }
    void setSinkPad(GRefPtr<GstPad>&&);

    GRefPtr<GstWebRTCRTPSender> sender() const { return m_sender; }
    GRefPtr<GstElement> bin() const { return m_bin; }

    bool configurePacketizers(GRefPtr<GstCaps>&&);

    GUniquePtr<GstStructure> parameters();
    void setInitialParameters(GUniquePtr<GstStructure>&&);
    void setParameters(GUniquePtr<GstStructure>&&);

    void configure(GRefPtr<GstCaps>&&);

    WARN_UNUSED_RETURN GUniquePtr<GstStructure> stats();

    virtual WARN_UNUSED_RETURN GRefPtr<GstPad> outgoingSourcePad() const = 0;
    virtual RefPtr<GStreamerRTPPacketizer> createPacketizer(RefPtr<UniqueSSRCGenerator>, const GstStructure*, GUniquePtr<GstStructure>&&) = 0;

    void replaceTrack(RefPtr<MediaStreamTrackPrivate>&&);

    virtual void teardown();

    virtual void dispatchBitrateRequest(uint32_t bitrate) = 0;

    RealtimeMediaSource::Type type() const;

protected:
    enum Type {
        Audio,
        Video
    };
    explicit RealtimeOutgoingMediaSourceGStreamer(Type, const RefPtr<UniqueSSRCGenerator>&, const String& mediaStreamId, MediaStreamTrack&);
    explicit RealtimeOutgoingMediaSourceGStreamer(Type, const RefPtr<UniqueSSRCGenerator>&);

    void initializeSourceFromTrackPrivate();
    virtual void sourceEnabledChanged();

    bool isStopped() const { return m_isStopped; }

    bool linkPacketizer(RefPtr<GStreamerRTPPacketizer>&&);

    Type m_type;
    String m_mediaStreamId;
    String m_trackId;
    String m_mid;

    bool m_enabled { true };
    bool m_muted { false };
    bool m_isStopped { true };
    RefPtr<MediaStreamTrackPrivate> m_track;
    std::optional<RealtimeMediaSourceSettings> m_initialSettings;
    GRefPtr<GstElement> m_bin;
    GRefPtr<GstElement> m_outgoingSource;
    GRefPtr<GstElement> m_preProcessor;
    GRefPtr<GstElement> m_tee;
    GRefPtr<GstElement> m_rtpFunnel;
    GRefPtr<GstElement> m_rtpCapsfilter;
    mutable GRefPtr<GstCaps> m_allowedCaps;
    GRefPtr<GstWebRTCRTPTransceiver> m_transceiver;
    GRefPtr<GstWebRTCRTPSender> m_sender;
    GRefPtr<GstPad> m_webrtcSinkPad;
    RefPtr<UniqueSSRCGenerator> m_ssrcGenerator;
    GUniquePtr<GstStructure> m_parameters;

    Vector<RefPtr<GStreamerRTPPacketizer>> m_packetizers;

private:
    void initialize();

    void sourceMutedChanged();

    void stopOutgoingSource();

    bool linkSource();
    virtual RTCRtpCapabilities rtpCapabilities() const = 0;
    void codecPreferencesChanged();

    // MediaStreamTrackPrivateObserver API
    void trackMutedChanged(MediaStreamTrackPrivate&) override { sourceMutedChanged(); }
    void trackEnabledChanged(MediaStreamTrackPrivate&) override { sourceEnabledChanged(); }
    void trackSettingsChanged(MediaStreamTrackPrivate&) override { initializeSourceFromTrackPrivate(); }
    void trackEnded(MediaStreamTrackPrivate&) override { }

    void checkMid();

    struct ExtensionLookupResults {
        bool hasRtpStreamIdExtension { false };
        bool hasRtpRepairedStreamIdExtension { false };
        bool hasMidExtension { false };
        int lastIdentifier { 0 };
    };
    ExtensionLookupResults lookupRtpExtensions(const GstStructure*);

    void startUpdatingStats();
    void stopUpdatingStats();

    RefPtr<GStreamerRTPPacketizer> getPacketizerForRid(StringView);
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
