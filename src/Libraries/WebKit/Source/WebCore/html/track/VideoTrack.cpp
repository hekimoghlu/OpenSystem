/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#include "VideoTrack.h"

#if ENABLE(VIDEO)

#include "CommonAtomStrings.h"
#include "ScriptExecutionContext.h"
#include "VideoTrackClient.h"
#include "VideoTrackConfiguration.h"
#include "VideoTrackList.h"
#include "VideoTrackPrivate.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MEDIA_SOURCE)
#include "SourceBuffer.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoTrack);

const AtomString& VideoTrack::signKeyword()
{
    static MainThreadNeverDestroyed<const AtomString> sign("sign"_s);
    return sign;
}

VideoTrack::VideoTrack(ScriptExecutionContext* context, VideoTrackPrivate& trackPrivate)
    : MediaTrackBase(context, MediaTrackBase::VideoTrack, trackPrivate.trackUID(), trackPrivate.id(), trackPrivate.label(), trackPrivate.language())
    , m_private(trackPrivate)
    , m_configuration(VideoTrackConfiguration::create())
    , m_selected(trackPrivate.selected())
{
    addClientToTrackPrivateBase(*this, trackPrivate);
    updateKindFromPrivate();
    updateConfigurationFromPrivate();
}

VideoTrack::~VideoTrack()
{
    removeClientFromTrackPrivateBase(Ref { m_private });
}

void VideoTrack::setPrivate(VideoTrackPrivate& trackPrivate)
{
    if (m_private.ptr() == &trackPrivate)
        return;

    removeClientFromTrackPrivateBase(Ref { m_private });
    m_private = trackPrivate;
    addClientToTrackPrivateBase(*this, trackPrivate);
#if !RELEASE_LOG_DISABLED
    m_private->setLogger(logger(), logIdentifier());
#endif

    m_private->setSelected(m_selected);
    updateKindFromPrivate();
    updateConfigurationFromPrivate();
    setId(m_private->id());
}

bool VideoTrack::isValidKind(const AtomString& value) const
{
    return value == "alternative"_s
        || value == "commentary"_s
        || value == "captions"_s
        || value == "main"_s
        || value == "sign"_s
        || value == "subtitles"_s;
}

void VideoTrack::setSelected(const bool selected)
{
    if (m_selected == selected)
        return;

    m_selected = selected;
    m_private->setSelected(selected);

    m_clients.forEach([this] (auto& client) {
        client.videoTrackSelectedChanged(*this);
    });
}

void VideoTrack::addClient(VideoTrackClient& client)
{
    ASSERT(!m_clients.contains(client));
    m_clients.add(client);
}

void VideoTrack::clearClient(VideoTrackClient& client)
{
    ASSERT(m_clients.contains(client));
    m_clients.remove(client);
}

size_t VideoTrack::inbandTrackIndex()
{
    return m_private->trackIndex();
}

void VideoTrack::selectedChanged(bool selected)
{
    setSelected(selected);
    m_clients.forEach([this] (auto& client) {
        client.videoTrackSelectedChanged(*this);
    });
}

void VideoTrack::configurationChanged(const PlatformVideoTrackConfiguration& configuration)
{
    m_configuration->setState(configuration);
}

void VideoTrack::idChanged(TrackID id)
{
    setId(id);
    m_clients.forEach([this] (auto& client) {
        client.videoTrackIdChanged(*this);
    });
}

void VideoTrack::labelChanged(const AtomString& label)
{
    setLabel(label);
    m_clients.forEach([this] (auto& client) {
        client.videoTrackLabelChanged(*this);
    });
}

void VideoTrack::languageChanged(const AtomString& language)
{
    setLanguage(language);
}

void VideoTrack::willRemove()
{
    m_clients.forEach([this] (auto& client) {
        client.willRemoveVideoTrack(*this);
    });
}

void VideoTrack::setKind(const AtomString& kind)
{
    // 10.1 kind, on setting:
    // 1. If the value being assigned to this attribute does not match one of the video track kinds,
    // then abort these steps.
    if (!isValidKind(kind))
        return;

    // 2. Update this attribute to the new value.
    setKindInternal(kind);

    // 3. If the sourceBuffer attribute on this track is not null, then queue a task to fire a simple
    // event named change at sourceBuffer.videoTracks.
    // 4. Queue a task to fire a simple event named change at the VideoTrackList object referenced by
    // the videoTracks attribute on the HTMLMediaElement.
    m_clients.forEach([this] (auto& client) {
        client.videoTrackKindChanged(*this);
    });
}

void VideoTrack::setLanguage(const AtomString& language)
{
    // 10.1 language, on setting:
    // 1. If the value being assigned to this attribute is not an empty string or a BCP 47 language
    // tag[BCP47], then abort these steps.
    // BCP 47 validation is done in TrackBase::setLanguage() which is
    // shared between all tracks that support setting language.

    // 2. Update this attribute to the new value.
    TrackBase::setLanguage(language);

    // 3. If the sourceBuffer attribute on this track is not null, then queue a task to fire a simple
    // event named change at sourceBuffer.videoTracks.
    // 4. Queue a task to fire a simple event named change at the VideoTrackList object referenced by
    // the videoTracks attribute on the HTMLMediaElement.
    m_clients.forEach([&] (auto& client) {
        client.videoTrackLanguageChanged(*this);
    });
}

void VideoTrack::updateKindFromPrivate()
{
    switch (m_private->kind()) {
    case VideoTrackPrivate::Kind::Alternative:
        setKind("alternative"_s);
        return;
    case VideoTrackPrivate::Kind::Captions:
        setKind("captions"_s);
        return;
    case VideoTrackPrivate::Kind::Main:
        setKind("main"_s);
        return;
    case VideoTrackPrivate::Kind::Sign:
        setKind("sign"_s);
        return;
    case VideoTrackPrivate::Kind::Subtitles:
        setKind("subtitles"_s);
        return;
    case VideoTrackPrivate::Kind::Commentary:
        setKind("commentary"_s);
        return;
    case VideoTrackPrivate::Kind::None:
        setKind(emptyAtom());
        return;
    }
    ASSERT_NOT_REACHED();
}

void VideoTrack::updateConfigurationFromPrivate()
{
    m_configuration->setState(m_private->configuration());
}

#if !RELEASE_LOG_DISABLED
void VideoTrack::setLogger(const Logger& logger, uint64_t logIdentifier)
{
    TrackBase::setLogger(logger, logIdentifier);
    m_private->setLogger(logger, this->logIdentifier());
}
#endif

} // namespace WebCore

#endif
