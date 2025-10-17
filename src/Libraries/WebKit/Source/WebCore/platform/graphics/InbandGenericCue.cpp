/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#include "InbandGenericCue.h"

#if ENABLE(VIDEO)

#include "ColorSerialization.h"


namespace WebCore {


InbandGenericCue::InbandGenericCue()
{
    m_cueData.m_uniqueId = InbandGenericCueIdentifier::generate();
}

String InbandGenericCue::toJSONString() const
{
    auto object = JSON::Object::create();

    object->setString("text"_s, m_cueData.m_content);
    object->setInteger("identifier"_s, m_cueData.m_uniqueId->toUInt64());
    object->setDouble("start"_s, m_cueData.m_startTime.toDouble());
    object->setDouble("end"_s, m_cueData.m_endTime.toDouble());

    ASCIILiteral status = ""_s;
    switch (m_cueData.m_status) {
    case GenericCueData::Status::Uninitialized:
        status = "Uninitialized"_s;
        break;
    case GenericCueData::Status::Partial:
        status = "Partial"_s;
        break;
    case GenericCueData::Status::Complete:
        status = "Complete"_s;
        break;
    }
    object->setString("status"_s, status);

    if (!m_cueData.m_id.isEmpty())
        object->setString("id"_s, m_cueData.m_id);

    if (m_cueData.m_line > 0)
        object->setDouble("line"_s, m_cueData.m_line);

    if (m_cueData.m_size > 0)
        object->setDouble("size"_s, m_cueData.m_size);

    if (m_cueData.m_position > 0)
        object->setDouble("position"_s, m_cueData.m_position);

    if (m_cueData.m_positionAlign != GenericCueData::Alignment::None) {
        ASCIILiteral positionAlign = ""_s;
        switch (m_cueData.m_positionAlign) {
        case GenericCueData::Alignment::Start:
            positionAlign = "Start"_s;
            break;
        case GenericCueData::Alignment::Middle:
            positionAlign = "Middle"_s;
            break;
        case GenericCueData::Alignment::End:
            positionAlign = "End"_s;
            break;
        case GenericCueData::Alignment::None:
            positionAlign = "None"_s;
            break;
        }
        object->setString("positionAlign"_s, positionAlign);
    }

    if (m_cueData.m_align != GenericCueData::Alignment::None) {
        ASCIILiteral align = ""_s;
        switch (m_cueData.m_align) {
        case GenericCueData::Alignment::Start:
            align = "Start"_s;
            break;
        case GenericCueData::Alignment::Middle:
            align = "Middle"_s;
            break;
        case GenericCueData::Alignment::End:
            align = "End"_s;
            break;
        case GenericCueData::Alignment::None:
            align = "None"_s;
            break;
        }
        object->setString("align"_s, align);
    }

    if (m_cueData.m_foregroundColor.isValid())
        object->setString("foregroundColor"_s, serializationForHTML(m_cueData.m_foregroundColor));

    if (m_cueData.m_backgroundColor.isValid())
        object->setString("backgroundColor"_s, serializationForHTML(m_cueData.m_backgroundColor));

    if (m_cueData.m_highlightColor.isValid())
        object->setString("highlightColor"_s, serializationForHTML(m_cueData.m_highlightColor));

    if (m_cueData.m_baseFontSize)
        object->setDouble("baseFontSize"_s, m_cueData.m_baseFontSize);

    if (m_cueData.m_relativeFontSize)
        object->setDouble("relativeFontSize"_s, m_cueData.m_relativeFontSize);

    if (!m_cueData.m_fontName.isEmpty())
        object->setString("font"_s, m_cueData.m_fontName);

    return object->toJSONString();
}

bool GenericCueData::equalNotConsideringTimesOrId(const GenericCueData& other) const
{
    return m_relativeFontSize == other.m_relativeFontSize
        && m_baseFontSize == other.m_baseFontSize
        && m_position == other.m_position
        && m_line == other.m_line
        && m_size == other.m_size
        && m_align == other.m_align
        && m_foregroundColor == other.m_foregroundColor
        && m_backgroundColor == other.m_backgroundColor
        && m_highlightColor == other.m_highlightColor
        && m_fontName == other.m_fontName
        && m_id == other.m_id
        && m_content == other.m_content;
}

}

#endif
