/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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
#include "InbandGenericTextTrack.h"

#if ENABLE(VIDEO)

#include "InbandTextTrackPrivate.h"
#include "Logging.h"
#include "ScriptExecutionContext.h"
#include "TextTrackCueList.h"
#include "TextTrackList.h"
#include "VTTRegionList.h"
#include <math.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(InbandGenericTextTrack);

void GenericTextTrackCueMap::add(InbandGenericCueIdentifier inbandCueIdentifier, TextTrackCueGeneric& publicCue)
{
    m_dataToCueMap.add(inbandCueIdentifier, &publicCue);
    m_cueToDataMap.add(&publicCue, inbandCueIdentifier);
}

TextTrackCueGeneric* GenericTextTrackCueMap::find(InbandGenericCueIdentifier inbandCueIdentifier)
{
    return m_dataToCueMap.get(inbandCueIdentifier);
}

void GenericTextTrackCueMap::remove(InbandGenericCueIdentifier inbandCueIdentifier)
{
    if (auto publicCue = m_dataToCueMap.take(inbandCueIdentifier))
        m_cueToDataMap.remove(publicCue.get());
}

void GenericTextTrackCueMap::remove(TextTrackCue& publicCue)
{
    if (auto cueIdentifier = m_cueToDataMap.takeOptional(&publicCue))
        m_dataToCueMap.remove(*cueIdentifier);
}

inline InbandGenericTextTrack::InbandGenericTextTrack(ScriptExecutionContext& context, InbandTextTrackPrivate& trackPrivate)
    : InbandTextTrack(context, trackPrivate)
{
}

Ref<InbandGenericTextTrack> InbandGenericTextTrack::create(ScriptExecutionContext& context, InbandTextTrackPrivate& trackPrivate)
{
    auto textTrack = adoptRef(*new InbandGenericTextTrack(context, trackPrivate));
    textTrack->suspendIfNeeded();
    return textTrack;
}

InbandGenericTextTrack::~InbandGenericTextTrack() = default;

void InbandGenericTextTrack::updateCueFromCueData(TextTrackCueGeneric& cue, InbandGenericCue& inbandCue)
{
    cue.willChange();

    cue.setStartTime(inbandCue.startTime());
    MediaTime endTime = inbandCue.endTime();
    if (endTime.isPositiveInfinite() && textTrackList() && textTrackList()->duration().isValid())
        endTime = textTrackList()->duration();
    cue.setEndTime(endTime);
    cue.setText(inbandCue.content());
    cue.setId(inbandCue.id());
    cue.setBaseFontSizeRelativeToVideoHeight(inbandCue.baseFontSize());
    cue.setFontSizeMultiplier(inbandCue.relativeFontSize());
    cue.setFontName(inbandCue.fontName());

    if (inbandCue.position() > 0)
        cue.setPosition(std::round(inbandCue.position()));
    if (inbandCue.line() > 0)
        cue.setLine(std::round(inbandCue.line()));
    if (inbandCue.size() > 0)
        cue.setSize(std::round(inbandCue.size()));
    if (inbandCue.backgroundColor().isValid())
        cue.setBackgroundColor(inbandCue.backgroundColor());
    if (inbandCue.foregroundColor().isValid())
        cue.setForegroundColor(inbandCue.foregroundColor());
    if (inbandCue.highlightColor().isValid())
        cue.setHighlightColor(inbandCue.highlightColor());

    switch (inbandCue.positionAlign()) {
    case GenericCueData::Alignment::Start:
        cue.setPositionAlign(VTTCue::PositionAlignSetting::LineLeft);
        break;
    case GenericCueData::Alignment::Middle:
        cue.setPositionAlign(VTTCue::PositionAlignSetting::Center);
        break;
    case GenericCueData::Alignment::End:
        cue.setPositionAlign(VTTCue::PositionAlignSetting::LineRight);
        break;
    case GenericCueData::Alignment::None:
        break;
    }

    if (inbandCue.align() == GenericCueData::Alignment::Start)
        cue.setAlign(VTTCue::AlignSetting::Start);
    else if (inbandCue.align() == GenericCueData::Alignment::Middle)
        cue.setAlign(VTTCue::AlignSetting::Center);
    else if (inbandCue.align() == GenericCueData::Alignment::End)
        cue.setAlign(VTTCue::AlignSetting::End);
    cue.setSnapToLines(false);

    cue.didChange();
}

void InbandGenericTextTrack::addGenericCue(InbandGenericCue& inbandCue)
{
    if (m_cueMap.find(inbandCue.uniqueId()))
        return;

    auto cue = TextTrackCueGeneric::create(*scriptExecutionContext(), inbandCue.startTime(), inbandCue.endTime(), inbandCue.content());
    updateCueFromCueData(cue.get(), inbandCue);
    if (hasCue(cue, TextTrackCue::IgnoreDuration)) {
        INFO_LOG(LOGIDENTIFIER, "ignoring already added cue: ", cue.get());
        return;
    }

    INFO_LOG(LOGIDENTIFIER, "added cue: ", cue.get());

    if (inbandCue.status() != GenericCueData::Status::Complete)
        m_cueMap.add(inbandCue.uniqueId(), cue);

    addCue(WTFMove(cue));
}

void InbandGenericTextTrack::updateGenericCue(InbandGenericCue& inbandCue)
{
    RefPtr cue = m_cueMap.find(inbandCue.uniqueId());
    if (!cue)
        return;

    updateCueFromCueData(*cue, inbandCue);

    if (inbandCue.status() == GenericCueData::Status::Complete)
        m_cueMap.remove(inbandCue.uniqueId());
}

void InbandGenericTextTrack::removeGenericCue(InbandGenericCue& inbandCue)
{
    RefPtr cue = m_cueMap.find(inbandCue.uniqueId());
    if (cue) {
        INFO_LOG(LOGIDENTIFIER, *cue);
        removeCue(*cue);
    } else
        INFO_LOG(LOGIDENTIFIER, "UNABLE to find cue: ", inbandCue);

}

ExceptionOr<void> InbandGenericTextTrack::removeCue(TextTrackCue& cue)
{
    auto result = TextTrack::removeCue(cue);
    if (!result.hasException())
        m_cueMap.remove(cue);
    return result;
}

WebVTTParser& InbandGenericTextTrack::parser()
{
    ASSERT(is<Document>(scriptExecutionContext()));
    if (!m_webVTTParser)
        m_webVTTParser = makeUnique<WebVTTParser>(static_cast<WebVTTParserClient&>(*this), downcast<Document>(*scriptExecutionContext()));
    return *m_webVTTParser;
}

void InbandGenericTextTrack::parseWebVTTCueData(ISOWebVTTCue&& cueData)
{
    parser().parseCueData(WTFMove(cueData));
}

void InbandGenericTextTrack::parseWebVTTFileHeader(String&& header)
{
    parser().parseFileHeader(WTFMove(header));
}

RefPtr<TextTrackCue> InbandGenericTextTrack::cueToExtend(TextTrackCue& newCue)
{
    if (newCue.startMediaTime() < MediaTime::zeroTime() || newCue.endMediaTime() < MediaTime::zeroTime())
        return nullptr;

    if (!m_cues || m_cues->length() < 2)
        return nullptr;

    return [this, &newCue]() -> RefPtr<TextTrackCue> {
        for (size_t i = 0; i < m_cues->length(); ++i) {
            auto existingCue = m_cues->item(i);
            ASSERT(existingCue->track() == this);

            if (abs(newCue.startMediaTime() - existingCue->startMediaTime()) > startTimeVariance())
                continue;

            if (abs(newCue.startMediaTime() - existingCue->endMediaTime()) > startTimeVariance())
                return nullptr;

            if (existingCue->cueContentsMatch(newCue))
                return existingCue;
        }

        return nullptr;
    }();
}

void InbandGenericTextTrack::newCuesParsed()
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document)
        return;

    for (auto& cueData : parser().takeCues()) {
        auto cue = VTTCue::create(*document, cueData);

        auto existingCue = cueToExtend(cue);
        if (!existingCue)
            existingCue = matchCue(cue, TextTrackCue::IgnoreDuration);

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

void InbandGenericTextTrack::newRegionsParsed()
{
    for (auto& region : parser().takeRegions())
        regions()->add(WTFMove(region));
}

void InbandGenericTextTrack::newStyleSheetsParsed()
{
    m_styleSheets = parser().takeStyleSheets();
}

void InbandGenericTextTrack::fileFailedToParse()
{
    ERROR_LOG(LOGIDENTIFIER);
}

} // namespace WebCore

#endif
