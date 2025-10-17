/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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

#include "BufferedLineReader.h"
#include "DocumentFragment.h"
#include "HTMLNames.h"
#include "TextResourceDecoder.h"
#include "VTTRegion.h"
#include <memory>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

using namespace HTMLNames;

class Document;
class ISOWebVTTCue;
class VTTScanner;

class WebVTTParserClient {
public:
    virtual ~WebVTTParserClient() = default;

    virtual void newCuesParsed() = 0;
    virtual void newRegionsParsed() = 0;
    virtual void newStyleSheetsParsed() = 0;
    virtual void fileFailedToParse() = 0;
};

class WebVTTCueData final : public RefCounted<WebVTTCueData> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(WebVTTCueData, WEBCORE_EXPORT);
public:

    static Ref<WebVTTCueData> create() { return adoptRef(*new WebVTTCueData()); }
    ~WebVTTCueData() = default;

    MediaTime startTime() const { return m_startTime; }
    void setStartTime(const MediaTime& startTime) { m_startTime = startTime; }

    MediaTime endTime() const { return m_endTime; }
    void setEndTime(const MediaTime& endTime) { m_endTime = endTime; }

    AtomString id() const { return m_id; }
    void setId(const AtomString& id) { m_id = id; }

    String content() const { return m_content; }
    void setContent(const String& content) { m_content = content; }

    String settings() const { return m_settings; }
    void setSettings(const String& settings) { m_settings = settings; }

    MediaTime originalStartTime() const { return m_originalStartTime; }
    void setOriginalStartTime(const MediaTime& time) { m_originalStartTime = time; }

private:
    WebVTTCueData() = default;

    MediaTime m_startTime;
    MediaTime m_endTime;
    MediaTime m_originalStartTime;
    AtomString m_id;
    String m_content;
    String m_settings;
};

class WEBCORE_EXPORT WebVTTParser final {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(WebVTTParser, WEBCORE_EXPORT);
public:
    enum ParseState {
        Initial,
        Header,
        Id,
        TimingsAndSettings,
        CueText,
        Region,
        Style,
        BadCue,
        Finished
    };

    WebVTTParser() = delete;
    WebVTTParser(WebVTTParserClient&, Document&);

    static inline bool isRecognizedTag(const AtomString& tagName)
    {
        return tagName == iTag
            || tagName == bTag
            || tagName == uTag
            || tagName == rubyTag
            || tagName == rtTag;
    }

    static bool collectTimeStamp(const String&, MediaTime&);

    // Useful functions for parsing percentage settings.
    static bool parseFloatPercentageValue(VTTScanner& valueScanner, float&);
    static bool parseFloatPercentageValuePair(VTTScanner& valueScanner, char, FloatPoint&);

    // Input data to the parser to parse.
    void parseBytes(std::span<const uint8_t>);
    void parseFileHeader(String&&);
    void parseCueData(const ISOWebVTTCue&);
    void flush();
    void fileFinished();

    // Transfers ownership of last parsed cues to caller.
    Vector<Ref<WebVTTCueData>> takeCues();
    Vector<Ref<VTTRegion>> takeRegions();
    Vector<String> takeStyleSheets();
    
    // Create the DocumentFragment representation of the WebVTT cue text.
    static Ref<DocumentFragment> createDocumentFragmentFromCueText(Document&, const String&);

private:
    void parse();
    void flushPendingCue();
    bool hasRequiredFileIdentifier(const String&);
    ParseState collectCueId(const String&);
    ParseState collectTimingsAndSettings(const String&);
    ParseState collectCueText(const String&);
    ParseState recoverCue(const String&);
    ParseState ignoreBadCue(const String&);
    ParseState collectRegionSettings(const String&);
    ParseState collectWebVTTBlock(const String&);
    ParseState checkAndRecoverCue(const String& line);
    ParseState collectStyleSheet(const String&);
    bool checkAndCreateRegion(StringView line);
    bool checkAndStoreRegion(StringView line);
    bool checkStyleSheet(StringView line);
    bool checkAndStoreStyleSheet(StringView line);

    void createNewCue();
    void resetCueValues();

    static bool collectTimeStamp(VTTScanner& input, MediaTime& timeStamp);

    Document& m_document;
    ParseState m_state { Initial };

    BufferedLineReader m_lineReader;
    RefPtr<TextResourceDecoder> m_decoder;
    AtomString m_currentId;
    MediaTime m_currentStartTime;
    MediaTime m_currentEndTime;
    StringBuilder m_currentContent;
    String m_previousLine;
    String m_currentSettings;
    RefPtr<VTTRegion> m_currentRegion;
    StringBuilder m_currentSourceStyleSheet;
    
    WebVTTParserClient& m_client;

    Vector<Ref<WebVTTCueData>> m_cuelist;
    Vector<Ref<VTTRegion>> m_regionList;
    Vector<String> m_styleSheets;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
