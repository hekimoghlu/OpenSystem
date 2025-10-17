/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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

#include "AudioTrackPrivateClient.h"
#include "PlatformAudioTrackConfiguration.h"
#include "TrackPrivateBase.h"
#include <wtf/Function.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(VIDEO)

namespace WebCore {

struct AudioInfo;

class AudioTrackPrivate : public TrackPrivateBase {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(AudioTrackPrivate);
public:
    virtual void setEnabled(bool enabled)
    {
        if (m_enabled == enabled)
            return;
        m_enabled = enabled;
        notifyClients([enabled](auto& client) {
            downcast<AudioTrackPrivateClient>(client).enabledChanged(enabled);
        });
        if (m_enabledChangedCallback)
            m_enabledChangedCallback(*this, m_enabled);
    }

    bool enabled() const { return m_enabled; }

    enum class Kind : uint8_t { Alternative, Description, Main, MainDesc, Translation, Commentary, None };
    virtual Kind kind() const { return Kind::None; }

    virtual bool isBackedByMediaStreamTrack() const { return false; }

    using EnabledChangedCallback = Function<void(AudioTrackPrivate&, bool enabled)>;
    void setEnabledChangedCallback(EnabledChangedCallback&& callback) { m_enabledChangedCallback = WTFMove(callback); }

    const PlatformAudioTrackConfiguration& configuration() const { return m_configuration; }
    void setConfiguration(PlatformAudioTrackConfiguration&& configuration)
    {
        if (configuration == m_configuration)
            return;
        m_configuration = WTFMove(configuration);
        notifyClients([configuration = m_configuration](auto& client) {
            downcast<AudioTrackPrivateClient>(client).configurationChanged(configuration);
        });
    }

    virtual void setFormatDescription(Ref<AudioInfo>&&) { }

    bool operator==(const AudioTrackPrivate& track) const
    {
        return TrackPrivateBase::operator==(track)
            && configuration() == track.configuration()
            && kind() == track.kind();
    }

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const override { return "AudioTrackPrivate"_s; }
#endif

    Type type() const final { return Type::Audio; }

protected:
    AudioTrackPrivate() = default;

private:
    bool m_enabled { false };
    PlatformAudioTrackConfiguration m_configuration;
    EnabledChangedCallback m_enabledChangedCallback;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioTrackPrivate)
static bool isType(const WebCore::TrackPrivateBase& track) { return track.type() == WebCore::TrackPrivateBase::Type::Audio; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
