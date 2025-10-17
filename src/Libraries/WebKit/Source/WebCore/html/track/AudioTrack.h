/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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

#if ENABLE(VIDEO)

#include "AudioTrackPrivateClient.h"
#include "TrackBase.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class AudioTrackClient;
class AudioTrackConfiguration;
class AudioTrackList;

class AudioTrack final : public MediaTrackBase, private AudioTrackPrivateClient {
    WTF_MAKE_TZONE_ALLOCATED(AudioTrack);
public:
    static Ref<AudioTrack> create(ScriptExecutionContext* context, AudioTrackPrivate& trackPrivate)
    {
        return adoptRef(*new AudioTrack(context, trackPrivate));
    }
    virtual ~AudioTrack();

    static const AtomString& descriptionKeyword();
    static const AtomString& mainDescKeyword();
    static const AtomString& translationKeyword();

    bool enabled() const final { return m_enabled; }
    void setEnabled(const bool);

    void addClient(AudioTrackClient&);
    void clearClient(AudioTrackClient&);

    size_t inbandTrackIndex() const;

    const AudioTrackPrivate& privateTrack() const { return m_private; }
    void setPrivate(AudioTrackPrivate&);

    void setLanguage(const AtomString&) final;

    AudioTrackConfiguration& configuration() const { return m_configuration; }

#if !RELEASE_LOG_DISABLED
    void setLogger(const Logger&, uint64_t) final;
#endif

private:
    AudioTrack(ScriptExecutionContext*, AudioTrackPrivate&);

    bool isValidKind(const AtomString&) const final;

    // AudioTrackPrivateClient
    void enabledChanged(bool) final;
    void configurationChanged(const PlatformAudioTrackConfiguration&) final;

    // TrackPrivateBaseClient
    void idChanged(TrackID) final;
    void labelChanged(const AtomString&) final;
    void languageChanged(const AtomString&) final;
    void willRemove() final;

    void updateKindFromPrivate();
    void updateConfigurationFromPrivate();

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "AudioTrack"_s; }
#endif

    WeakPtr<AudioTrackList> m_audioTrackList;
    WeakHashSet<AudioTrackClient> m_clients;
    Ref<AudioTrackPrivate> m_private;
    bool m_enabled { false };

    Ref<AudioTrackConfiguration> m_configuration;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioTrack)
    static bool isType(const WebCore::TrackBase& track) { return track.type() == WebCore::TrackBase::AudioTrack; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
