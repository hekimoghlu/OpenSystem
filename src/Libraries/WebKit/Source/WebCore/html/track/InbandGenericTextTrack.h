/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

#include "InbandTextTrack.h"
#include "TextTrackCueGeneric.h"
#include "WebVTTParser.h"

namespace WebCore {

class GenericTextTrackCueMap {
public:
    void add(InbandGenericCueIdentifier, TextTrackCueGeneric&);

    void remove(TextTrackCue&);
    void remove(InbandGenericCueIdentifier);

    TextTrackCueGeneric* find(InbandGenericCueIdentifier);

private:
    using CueToDataMap = HashMap<TextTrackCue*, InbandGenericCueIdentifier>;
    using CueDataToCueMap = HashMap<InbandGenericCueIdentifier, RefPtr<TextTrackCueGeneric>>;

    CueToDataMap m_cueToDataMap;
    CueDataToCueMap m_dataToCueMap;
};

class InbandGenericTextTrack final : public InbandTextTrack, private WebVTTParserClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InbandGenericTextTrack);
public:
    static Ref<InbandGenericTextTrack> create(ScriptExecutionContext&, InbandTextTrackPrivate&);
    virtual ~InbandGenericTextTrack();

private:
    InbandGenericTextTrack(ScriptExecutionContext&, InbandTextTrackPrivate&);

    void addGenericCue(InbandGenericCue&) final;
    void updateGenericCue(InbandGenericCue&) final;
    void removeGenericCue(InbandGenericCue&) final;
    ExceptionOr<void> removeCue(TextTrackCue&) final;

    void updateCueFromCueData(TextTrackCueGeneric&, InbandGenericCue&);

    RefPtr<TextTrackCue> cueToExtend(TextTrackCue&);

    WebVTTParser& parser();
    void parseWebVTTCueData(ISOWebVTTCue&&) final;
    void parseWebVTTFileHeader(String&&) final;

    void newCuesParsed() final;
    void newRegionsParsed() final;
    void newStyleSheetsParsed() final;
    void fileFailedToParse() final;

    bool shouldPurgeCuesFromUnbufferedRanges() const final { return true; }

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "InbandGenericTextTrack"_s; }
#endif

    GenericTextTrackCueMap m_cueMap;
    std::unique_ptr<WebVTTParser> m_webVTTParser;
};

} // namespace WebCore

#endif
