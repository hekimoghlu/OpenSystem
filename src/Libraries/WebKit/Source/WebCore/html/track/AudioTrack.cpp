/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#include "AudioTrack.h"

#if ENABLE(VIDEO)

#include "AudioTrackClient.h"
#include "AudioTrackConfiguration.h"
#include "AudioTrackList.h"
#include "AudioTrackPrivate.h"
#include "CommonAtomStrings.h"
#include "ScriptExecutionContext.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioTrack);

const AtomString& AudioTrack::descriptionKeyword()
{
    static MainThreadNeverDestroyed<const AtomString> description("description"_s);
    return description;
}

const AtomString& AudioTrack::mainDescKeyword()
{
    static MainThreadNeverDestroyed<const AtomString> mainDesc("main-desc"_s);
    return mainDesc;
}

const AtomString& AudioTrack::translationKeyword()
{
    static MainThreadNeverDestroyed<const AtomString> translation("translation"_s);
    return translation;
}

AudioTrack::AudioTrack(ScriptExecutionContext* context, AudioTrackPrivate& trackPrivate)
    : MediaTrackBase(context, MediaTrackBase::AudioTrack, trackPrivate.trackUID(), trackPrivate.id(), trackPrivate.label(), trackPrivate.language())
    , m_private(trackPrivate)
    , m_enabled(trackPrivate.enabled())
    , m_configuration(AudioTrackConfiguration::create())
{
    addClientToTrackPrivateBase(*this, trackPrivate);

    updateKindFromPrivate();
    updateConfigurationFromPrivate();
}

AudioTrack::~AudioTrack()
{
    removeClientFromTrackPrivateBase(Ref { m_private });
}

void AudioTrack::setPrivate(AudioTrackPrivate& trackPrivate)
{
    if (m_private.ptr() == &trackPrivate)
        return;

    removeClientFromTrackPrivateBase(Ref { m_private });
    m_private = trackPrivate;
    m_private->setEnabled(m_enabled);
    addClientToTrackPrivateBase(*this, trackPrivate);

#if !RELEASE_LOG_DISABLED
    m_private->setLogger(logger(), logIdentifier());
#endif

    updateKindFromPrivate();
    updateConfigurationFromPrivate();
    setId(m_private->id());
}

void AudioTrack::setLanguage(const AtomString& language)
{
    TrackBase::setLanguage(language);

    m_clients.forEach([&] (auto& client) {
        client.audioTrackLanguageChanged(*this);
    });
}

bool AudioTrack::isValidKind(const AtomString& value) const
{
    return value == "alternative"_s
        || value == "commentary"_s
        || value == "description"_s
        || value == "main"_s
        || value == "main-desc"_s
        || value == "translation"_s;
}

void AudioTrack::setEnabled(bool enabled)
{
    if (m_enabled == enabled)
        return;

    m_private->setEnabled(enabled);
    m_clients.forEach([this] (auto& client) {
        client.audioTrackEnabledChanged(*this);
    });
}

void AudioTrack::addClient(AudioTrackClient& client)
{
    ASSERT(!m_clients.contains(client));
    m_clients.add(client);
}

void AudioTrack::clearClient(AudioTrackClient& client)
{
    ASSERT(m_clients.contains(client));
    m_clients.remove(client);
}

size_t AudioTrack::inbandTrackIndex() const
{
    return m_private->trackIndex();
}

void AudioTrack::enabledChanged(bool enabled)
{
    if (m_enabled == enabled)
        return;

    m_enabled = enabled;

    m_clients.forEach([this] (auto& client) {
        client.audioTrackEnabledChanged(*this);
    });
}

void AudioTrack::configurationChanged(const PlatformAudioTrackConfiguration& configuration)
{
    m_configuration->setState(configuration);
}

void AudioTrack::idChanged(TrackID id)
{
    setId(id);
    m_clients.forEach([this] (auto& client) {
        client.audioTrackIdChanged(*this);
    });
}

void AudioTrack::labelChanged(const AtomString& label)
{
    setLabel(label);
    m_clients.forEach([this] (auto& client) {
        client.audioTrackLabelChanged(*this);
    });
}

void AudioTrack::languageChanged(const AtomString& language)
{
    setLanguage(language);
}

void AudioTrack::willRemove()
{
    m_clients.forEach([this] (auto& client) {
        client.willRemoveAudioTrack(*this);
    });
}

void AudioTrack::updateKindFromPrivate()
{
    switch (m_private->kind()) {
    case AudioTrackPrivate::Kind::Alternative:
        setKind("alternative"_s);
        break;
    case AudioTrackPrivate::Kind::Description:
        setKind("description"_s);
        break;
    case AudioTrackPrivate::Kind::Main:
        setKind("main"_s);
        break;
    case AudioTrackPrivate::Kind::MainDesc:
        setKind("main-desc"_s);
        break;
    case AudioTrackPrivate::Kind::Translation:
        setKind("translation"_s);
        break;
    case AudioTrackPrivate::Kind::Commentary:
        setKind("commentary"_s);
        break;
    case AudioTrackPrivate::Kind::None:
        setKind(emptyAtom());
        break;
    default:
        ASSERT_NOT_REACHED();
        break;
    }
}

void AudioTrack::updateConfigurationFromPrivate()
{
    m_configuration->setState(m_private->configuration());
}

#if !RELEASE_LOG_DISABLED
void AudioTrack::setLogger(const Logger& logger, uint64_t logIdentifier)
{
    TrackBase::setLogger(logger, logIdentifier);
    m_private->setLogger(logger, this->logIdentifier());
}
#endif

} // namespace WebCore

#endif
