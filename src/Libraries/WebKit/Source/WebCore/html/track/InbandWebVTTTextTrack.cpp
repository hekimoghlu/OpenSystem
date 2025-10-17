/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#include "InbandWebVTTTextTrack.h"

#if ENABLE(VIDEO)

#include "InbandTextTrackPrivate.h"
#include "Logging.h"
#include "ScriptExecutionContext.h"
#include "VTTCue.h"
#include "VTTRegionList.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(InbandWebVTTTextTrack);

inline InbandWebVTTTextTrack::InbandWebVTTTextTrack(ScriptExecutionContext& context, InbandTextTrackPrivate& trackPrivate)
    : InbandTextTrack(context, trackPrivate)
{
}

Ref<InbandTextTrack> InbandWebVTTTextTrack::create(ScriptExecutionContext& context, InbandTextTrackPrivate& trackPrivate)
{
    auto textTrack = adoptRef(*new InbandWebVTTTextTrack(context, trackPrivate));
    textTrack->suspendIfNeeded();
    return textTrack;
}

InbandWebVTTTextTrack::~InbandWebVTTTextTrack() = default;

WebVTTParser& InbandWebVTTTextTrack::parser()
{
    ASSERT(is<Document>(scriptExecutionContext()));
    if (!m_webVTTParser)
        m_webVTTParser = makeUnique<WebVTTParser>(static_cast<WebVTTParserClient&>(*this), downcast<Document>(*scriptExecutionContext()));
    return *m_webVTTParser;
}

void InbandWebVTTTextTrack::parseWebVTTCueData(std::span<const uint8_t> data)
{
    parser().parseBytes(data);
}

void InbandWebVTTTextTrack::parseWebVTTCueData(ISOWebVTTCue&& cueData)
{
    parser().parseCueData(WTFMove(cueData));
}

void InbandWebVTTTextTrack::newCuesParsed()
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document)
        return;

    for (auto& cueData : parser().takeCues()) {
        auto cue = VTTCue::create(*document, cueData);
        auto existingCue = matchCue(cue, TextTrackCue::IgnoreDuration);
        if (!existingCue) {
            INFO_LOG(LOGIDENTIFIER, cue.get());
            addCue(WTFMove(cue));
            continue;
        }

        if (existingCue->endTime() >= cue->endTime()) {
            INFO_LOG(LOGIDENTIFIER, "ignoring already added cue: ", cue.get());
            continue;
        }

        ALWAYS_LOG(LOGIDENTIFIER, "extending endTime of existing cue: ", *existingCue, " to ", cue->endTime());
        existingCue->setEndTime(cue->endTime());
    }
}
    
void InbandWebVTTTextTrack::newRegionsParsed()
{
    for (auto& region : parser().takeRegions())
        regions()->add(WTFMove(region));
}

void InbandWebVTTTextTrack::newStyleSheetsParsed()
{
}

void InbandWebVTTTextTrack::fileFailedToParse()
{
    ERROR_LOG(LOGIDENTIFIER, "Error parsing WebVTT stream.");
}

} // namespace WebCore

#endif
