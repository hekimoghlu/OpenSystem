/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#include "ISOVTTCue.h"

#include "Logging.h"
#include <JavaScriptCore/DataView.h>
#include <wtf/JSONValues.h>
#include <wtf/URL.h>

using JSC::DataView;

namespace WebCore {

class ISOStringBox final : public ISOBox {
public:
    const String& contents() { return m_contents; }

private:
    bool parse(JSC::DataView& view, unsigned& offset) override
    {
        unsigned localOffset = offset;
        if (!ISOBox::parse(view, localOffset))
            return false;

        auto characterCount = m_size - (localOffset - offset);
        if (!characterCount) {
            m_contents = emptyString();
            return true;
        }

        auto bytesRemaining = view.byteLength() - localOffset;
        if (characterCount > bytesRemaining)
            return false;

        Vector<LChar> characters;
        characters.reserveInitialCapacity(static_cast<size_t>(characterCount));
        while (characterCount--) {
            int8_t character = 0;
            if (!checkedRead<int8_t>(character, view, localOffset, BigEndian))
                return false;
            characters.append(character);
        }

        m_contents = String::fromUTF8(characters);
        offset = localOffset;
        return true;
    }
    String m_contents;
};

static FourCC vttIdBoxType() { return std::span { "iden" }; }
static FourCC vttSettingsBoxType() { return std::span { "sttg" }; }
static FourCC vttPayloadBoxType() { return std::span { "payl" }; }
static FourCC vttCurrentTimeBoxType() { return std::span { "ctim" }; }
static FourCC vttCueSourceIDBoxType() { return std::span { "vsid" }; }

ISOWebVTTCue::ISOWebVTTCue(const MediaTime& presentationTime, const MediaTime& duration)
    : m_presentationTime(presentationTime)
    , m_duration(duration)
{
}

ISOWebVTTCue::ISOWebVTTCue(MediaTime&& presentationTime, MediaTime&& duration, AtomString&& cueID, String&& cueText, String&& settings, String&& sourceID, String&& originalStartTime)
    : m_presentationTime(WTFMove(presentationTime))
    , m_duration(WTFMove(duration))
    , m_sourceID(WTFMove(sourceID))
    , m_identifier(WTFMove(cueID))
    , m_originalStartTime(WTFMove(originalStartTime))
    , m_settings(WTFMove(settings))
    , m_cueText(WTFMove(cueText))
{
}

ISOWebVTTCue::ISOWebVTTCue() = default;
ISOWebVTTCue::ISOWebVTTCue(ISOWebVTTCue&&) = default;
ISOWebVTTCue::~ISOWebVTTCue() = default;

bool ISOWebVTTCue::parse(DataView& view, unsigned& offset)
{
    if (!ISOBox::parse(view, offset))
        return false;

    ISOStringBox stringBox;

    while (stringBox.read(view, offset)) {
        if (stringBox.boxType() == vttCueSourceIDBoxType())
            m_sourceID = stringBox.contents();
        else if (stringBox.boxType() == vttIdBoxType())
            m_identifier = AtomString { stringBox.contents() };
        else if (stringBox.boxType() == vttCurrentTimeBoxType())
            m_originalStartTime = stringBox.contents();
        else if (stringBox.boxType() == vttSettingsBoxType())
            m_settings = stringBox.contents();
        else if (stringBox.boxType() == vttPayloadBoxType())
            m_cueText = stringBox.contents();
        else
            LOG(Media, "ISOWebVTTCue::ISOWebVTTCue - skipping box id = \"%s\", size = %" PRIu64, stringBox.boxType().string().data(), stringBox.size());
    }
    return true;
}

String ISOWebVTTCue::toJSONString() const
{
    auto object = JSON::Object::create();

    object->setString("text"_s, m_cueText);
    object->setString("sourceId"_s, encodeWithURLEscapeSequences(m_sourceID));
    object->setString("id"_s, encodeWithURLEscapeSequences(m_identifier));

    object->setString("originalStartTime"_s, encodeWithURLEscapeSequences(m_originalStartTime));
    object->setString("settings"_s, encodeWithURLEscapeSequences(m_settings));

    object->setDouble("presentationTime"_s, m_presentationTime.toDouble());
    object->setDouble("duration"_s, m_duration.toDouble());

    return object->toJSONString();
}

} // namespace WebCore
