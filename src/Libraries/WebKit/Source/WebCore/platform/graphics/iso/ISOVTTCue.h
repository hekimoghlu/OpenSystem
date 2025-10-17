/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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

#include "ISOBox.h"
#include <wtf/MediaTime.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// 4 bytes : 4CC : identifier = 'vttc'
// 4 bytes : unsigned : length
// N bytes : CueSourceIDBox : box : optional
// N bytes : CueIDBox : box : optional
// N bytes : CueTimeBox : box : optional
// N bytes : CueSettingsBox : box : optional
// N bytes : CuePayloadBox : box : required

class ISOWebVTTCue final : public ISOBox {
public:
    ISOWebVTTCue(const MediaTime& presentationTime, const MediaTime& duration);
    WEBCORE_EXPORT ISOWebVTTCue(MediaTime&& presentationTime, MediaTime&& duration, AtomString&& cueID, String&& cueText, String&& settings = { }, String&& sourceID = { }, String&& originalStartTime = { });
    ISOWebVTTCue(const ISOWebVTTCue&) = default;
    WEBCORE_EXPORT ISOWebVTTCue();
    WEBCORE_EXPORT ISOWebVTTCue(ISOWebVTTCue&&);
    WEBCORE_EXPORT ~ISOWebVTTCue();

    ISOWebVTTCue& operator=(const ISOWebVTTCue&) = default;
    ISOWebVTTCue& operator=(ISOWebVTTCue&&) = default;

    static FourCC boxTypeName() { return std::span { "vttc" }; }

    const MediaTime& presentationTime() const { return m_presentationTime; }
    const MediaTime& duration() const { return m_duration; }

    const String& sourceID() const { return m_sourceID; }
    const AtomString& id() const { return m_identifier; }
    const String& originalStartTime() const { return m_originalStartTime; }
    const String& settings() const { return m_settings; }
    const String& cueText() const { return m_cueText; }

    String toJSONString() const;

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) override;

private:
    MediaTime m_presentationTime;
    MediaTime m_duration;

    String m_sourceID;
    AtomString m_identifier;
    String m_originalStartTime;
    String m_settings;
    String m_cueText;
};

} // namespace WebCore

namespace WTF {

template<typename Type>
struct LogArgument;

template <>
struct LogArgument<WebCore::ISOWebVTTCue> {
    static String toString(const WebCore::ISOWebVTTCue& cue)
    {
        return cue.toJSONString();
    }
};

} // namespace WTF
