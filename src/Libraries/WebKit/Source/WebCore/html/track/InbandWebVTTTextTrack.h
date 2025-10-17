/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#include "WebVTTParser.h"
#include <memory>

namespace WebCore {

class InbandWebVTTTextTrack final : public InbandTextTrack, private WebVTTParserClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InbandWebVTTTextTrack);
public:
    static Ref<InbandTextTrack> create(ScriptExecutionContext&, InbandTextTrackPrivate&);
    virtual ~InbandWebVTTTextTrack();

private:
    InbandWebVTTTextTrack(ScriptExecutionContext&, InbandTextTrackPrivate&);

    WebVTTParser& parser();
    void parseWebVTTCueData(std::span<const uint8_t>) final;
    void parseWebVTTCueData(ISOWebVTTCue&&) final;

    void newCuesParsed() final;
    void newRegionsParsed() final;
    void newStyleSheetsParsed() final;
    void fileFailedToParse() final;

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "InbandWebVTTTextTrack"_s; }
#endif

    std::unique_ptr<WebVTTParser> m_webVTTParser;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
