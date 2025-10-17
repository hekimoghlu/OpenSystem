/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#include "InbandDataTextTrack.h"

#if ENABLE(VIDEO)

#include "DataCue.h"
#include "InbandTextTrackPrivate.h"
#include "TextTrackList.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(InbandDataTextTrack);

inline InbandDataTextTrack::InbandDataTextTrack(ScriptExecutionContext& context, InbandTextTrackPrivate& trackPrivate)
    : InbandTextTrack(context, trackPrivate)
{
}

Ref<InbandDataTextTrack> InbandDataTextTrack::create(ScriptExecutionContext& context, InbandTextTrackPrivate& trackPrivate)
{
    auto textTrack = adoptRef(*new InbandDataTextTrack(context, trackPrivate));
    textTrack->suspendIfNeeded();
    return textTrack;
}

InbandDataTextTrack::~InbandDataTextTrack() = default;

void InbandDataTextTrack::addDataCue(const MediaTime& start, const MediaTime& end, std::span<const uint8_t> data)
{
    // FIXME: handle datacue creation on worker.
    if (RefPtr document = dynamicDowncast<Document>(scriptExecutionContext()))
        addCue(DataCue::create(*document, start, end, data));
}

#if ENABLE(DATACUE_VALUE)

void InbandDataTextTrack::addDataCue(const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&& platformValue, const String& type)
{
    // FIXME: handle datacue creation on worker.
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document)
        return;

    if (findIncompleteCue(platformValue))
        return;

    auto cue = DataCue::create(*document, start, end, platformValue.copyRef(), type);
    if (hasCue(cue, TextTrackCue::IgnoreDuration)) {
        INFO_LOG(LOGIDENTIFIER, "ignoring already added cue: ", cue.get());
        return;
    }

    auto* textTrackList = downcast<TextTrackList>(trackList());
    if (end.isPositiveInfinite()) {
        if (textTrackList && textTrackList->duration().isValid())
            cue->setEndTime(textTrackList->duration());
        m_incompleteCueMap.append(&cue.get());
    }

    INFO_LOG(LOGIDENTIFIER, cue.get());

    addCue(WTFMove(cue));
}

RefPtr<DataCue> InbandDataTextTrack::findIncompleteCue(const SerializedPlatformDataCue& cueToFind)
{
    auto index = m_incompleteCueMap.findIf([&](const auto& cue) {
        return cueToFind.isEqual(*cue->platformValue());
    });

    if (index == notFound)
        return nullptr;

    return m_incompleteCueMap[index];
}

void InbandDataTextTrack::updateDataCue(const MediaTime& start, const MediaTime& inEnd, SerializedPlatformDataCue& platformValue)
{
    auto cue = findIncompleteCue(platformValue);
    if (!cue)
        return;

    cue->willChange();

    MediaTime end = inEnd;
    auto* textTrackList = downcast<TextTrackList>(trackList());
    if (end.isPositiveInfinite() && textTrackList && textTrackList->duration().isValid())
        end = textTrackList->duration();
    else
        m_incompleteCueMap.removeFirst(cue);

    INFO_LOG(LOGIDENTIFIER, "was start = ", cue->startMediaTime(), ", end = ", cue->endMediaTime(), ", will be start = ", start, ", end = ", end);

    cue->setStartTime(start);
    cue->setEndTime(end);

    cue->didChange();
}

void InbandDataTextTrack::removeDataCue(const MediaTime&, const MediaTime&, SerializedPlatformDataCue& platformValue)
{
    if (auto cue = findIncompleteCue(platformValue)) {
        INFO_LOG(LOGIDENTIFIER, "removing: ", *cue);
        m_incompleteCueMap.removeFirst(cue);
        InbandTextTrack::removeCue(*cue);
    }
}

ExceptionOr<void> InbandDataTextTrack::removeCue(TextTrackCue& cue)
{
    ASSERT(cue.cueType() == TextTrackCue::Data);

    if (auto platformValue = const_cast<SerializedPlatformDataCue*>(downcast<DataCue>(cue).platformValue()))
        removeDataCue({ }, { }, *platformValue);

    return InbandTextTrack::removeCue(cue);
}

#endif

} // namespace WebCore

#endif
