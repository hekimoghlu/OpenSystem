/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include "InbandMetadataTextTrackPrivateAVF.h"

#if ENABLE(VIDEO) && ENABLE(DATACUE_VALUE) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))

#include "InbandTextTrackPrivateClient.h"
#include "Logging.h"
#include "MediaPlayer.h"
#include <CoreMedia/CoreMedia.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InbandMetadataTextTrackPrivateAVF);

Ref<InbandMetadataTextTrackPrivateAVF> InbandMetadataTextTrackPrivateAVF::create(InbandTextTrackPrivate::Kind kind, TrackID id, InbandTextTrackPrivate::CueFormat cueFormat)
{
    return adoptRef(*new InbandMetadataTextTrackPrivateAVF(kind, id, cueFormat));
}

InbandMetadataTextTrackPrivateAVF::InbandMetadataTextTrackPrivateAVF(InbandTextTrackPrivate::Kind kind, TrackID id, InbandTextTrackPrivate::CueFormat cueFormat)
    : InbandTextTrackPrivate(cueFormat)
    , m_kind(kind)
    , m_id(id)
{
}

InbandMetadataTextTrackPrivateAVF::~InbandMetadataTextTrackPrivateAVF() = default;

#if ENABLE(DATACUE_VALUE)

void InbandMetadataTextTrackPrivateAVF::addDataCue(const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&& cueData, const String& type)
{
    ASSERT(isMainThread());
    ASSERT(cueFormat() == CueFormat::Data);
    ASSERT(start >= MediaTime::zeroTime());

    if (!hasClients())
        return;
    ASSERT(hasOneClient());

    m_currentCueStartTime = start;
    if (end.isPositiveInfinite())
        m_incompleteCues.append(IncompleteMetaDataCue { cueData.ptr(), start });
    notifyMainThreadClient([&](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).addDataCue(start, end, WTFMove(cueData), type);
    });
}

void InbandMetadataTextTrackPrivateAVF::updatePendingCueEndTimes(const MediaTime& time)
{
    ASSERT(isMainThread());
    ASSERT(time >= MediaTime::zeroTime());

    if (time >= m_currentCueStartTime) {
        if (hasClients()) {
            ASSERT(hasOneClient());
            for (auto& partialCue : m_incompleteCues) {
                INFO_LOG(LOGIDENTIFIER, "updating cue: start = ", partialCue.startTime, ", end = ", time);
                notifyMainThreadClient([&](auto& client) {
                    downcast<InbandTextTrackPrivateClient>(client).updateDataCue(partialCue.startTime, time, *partialCue.cueData);
                });
            }
        }
    } else
        WARNING_LOG(LOGIDENTIFIER, "negative length cue(s) ignored: start = ", m_currentCueStartTime, ", end = ", time);

    m_incompleteCues.shrink(0);
    m_currentCueStartTime = MediaTime::zeroTime();
}

#endif

void InbandMetadataTextTrackPrivateAVF::flushPartialCues()
{
    ASSERT(isMainThread());
    if (m_currentCueStartTime && m_incompleteCues.size())
        INFO_LOG(LOGIDENTIFIER, "flushing incomplete data for cues: start = ", m_currentCueStartTime);

    if (hasClients()) {
        ASSERT(hasOneClient());
        for (auto& partialCue : m_incompleteCues) {
            notifyMainThreadClient([&](TrackPrivateBaseClient& client) {
                downcast<InbandTextTrackPrivateClient>(client).removeDataCue(partialCue.startTime, MediaTime::positiveInfiniteTime(), *partialCue.cueData);
            });
        }
    }

    m_incompleteCues.shrink(0);
    m_currentCueStartTime = MediaTime::zeroTime();
}

} // namespace WebCore

#endif // ENABLE(VIDEO) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))
