/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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
#include "InbandTextTrack.h"

#if ENABLE(VIDEO)

#include "InbandDataTextTrack.h"
#include "InbandGenericTextTrack.h"
#include "InbandTextTrackPrivate.h"
#include "InbandWebVTTTextTrack.h"
#include "ScriptExecutionContext.h"
#include "TextTrackClient.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(InbandTextTrack);

Ref<InbandTextTrack> InbandTextTrack::create(ScriptExecutionContext& context, InbandTextTrackPrivate& trackPrivate)
{
    switch (trackPrivate.cueFormat()) {
    case InbandTextTrackPrivate::CueFormat::Data:
        return InbandDataTextTrack::create(context, trackPrivate);
    case InbandTextTrackPrivate::CueFormat::Generic:
        return InbandGenericTextTrack::create(context, trackPrivate);
    case InbandTextTrackPrivate::CueFormat::WebVTT:
        return InbandWebVTTTextTrack::create(context, trackPrivate);
    }
    ASSERT_NOT_REACHED();
    auto textTrack = InbandDataTextTrack::create(context, trackPrivate);
    textTrack->suspendIfNeeded();
    return textTrack;
}

InbandTextTrack::InbandTextTrack(ScriptExecutionContext& context, InbandTextTrackPrivate& trackPrivate)
    : TextTrack(&context, emptyAtom(), trackPrivate.id(), trackPrivate.label(), trackPrivate.language(), InBand)
    , m_private(trackPrivate)
{
    addClientToTrackPrivateBase(*this, trackPrivate);
    updateKindFromPrivate();
}

InbandTextTrack::~InbandTextTrack()
{
    removeClientFromTrackPrivateBase(Ref { m_private });
}

void InbandTextTrack::setPrivate(InbandTextTrackPrivate& trackPrivate)
{
    if (m_private.ptr() == &trackPrivate)
        return;

    removeClientFromTrackPrivateBase(Ref { m_private });
    m_private = trackPrivate;
    addClientToTrackPrivateBase(*this, trackPrivate);

    setModeInternal(mode());
    updateKindFromPrivate();
    setId(m_private->id());
}

void InbandTextTrack::setMode(Mode mode)
{
    TextTrack::setMode(mode);
    setModeInternal(mode);
}

static inline InbandTextTrackPrivate::Mode toPrivate(TextTrack::Mode mode)
{
    switch (mode) {
    case TextTrack::Mode::Disabled:
        return InbandTextTrackPrivate::Mode::Disabled;
    case TextTrack::Mode::Hidden:
        return InbandTextTrackPrivate::Mode::Hidden;
    case TextTrack::Mode::Showing:
        return InbandTextTrackPrivate::Mode::Showing;
    }
    ASSERT_NOT_REACHED();
    return InbandTextTrackPrivate::Mode::Disabled;
}

void InbandTextTrack::setModeInternal(Mode mode)
{
    m_private->setMode(toPrivate(mode));
}

bool InbandTextTrack::isClosedCaptions() const
{
    return m_private->isClosedCaptions();
}

bool InbandTextTrack::isSDH() const
{
    return m_private->isSDH();
}

bool InbandTextTrack::containsOnlyForcedSubtitles() const
{
    return m_private->containsOnlyForcedSubtitles();
}

bool InbandTextTrack::isMainProgramContent() const
{
    return m_private->isMainProgramContent();
}

bool InbandTextTrack::isEasyToRead() const
{
    return m_private->isEasyToRead();
}

bool InbandTextTrack::isDefault() const
{
    return m_private->isDefault();
}

size_t InbandTextTrack::inbandTrackIndex()
{
    return m_private->trackIndex();
}

AtomString InbandTextTrack::inBandMetadataTrackDispatchType() const
{
    return m_private->inBandMetadataTrackDispatchType();
}

void InbandTextTrack::idChanged(TrackID id)
{
    setId(id);
}

void InbandTextTrack::labelChanged(const AtomString& label)
{
    setLabel(label);
}

void InbandTextTrack::languageChanged(const AtomString& language)
{
    setLanguage(language);
}

void InbandTextTrack::willRemove()
{
    m_clients.forEach([&] (auto& client) {
        client.willRemoveTextTrack(*this);
    });
}

void InbandTextTrack::updateKindFromPrivate()
{
    switch (m_private->kind()) {
    case InbandTextTrackPrivate::Kind::Subtitles:
        setKind(Kind::Subtitles);
        return;
    case InbandTextTrackPrivate::Kind::Captions:
        setKind(Kind::Captions);
        return;
    case InbandTextTrackPrivate::Kind::Descriptions:
        setKind(Kind::Descriptions);
        return;
    case InbandTextTrackPrivate::Kind::Chapters:
        setKind(Kind::Chapters);
        return;
    case InbandTextTrackPrivate::Kind::Metadata:
        setKind(Kind::Metadata);
        return;
    case InbandTextTrackPrivate::Kind::Forced:
        setKind(Kind::Forced);
        return;
    case InbandTextTrackPrivate::Kind::None:
        break;
    }
    ASSERT_NOT_REACHED();
}

MediaTime InbandTextTrack::startTimeVariance() const
{
    return m_private->startTimeVariance();
}

#if !RELEASE_LOG_DISABLED
void InbandTextTrack::setLogger(const Logger& logger, uint64_t logIdentifier)
{
    TextTrack::setLogger(logger, logIdentifier);
    m_private->setLogger(logger, this->logIdentifier());
}
#endif

} // namespace WebCore

#endif
