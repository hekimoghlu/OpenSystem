/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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

#include "Color.h"
#include "InbandGenericCueIdentifier.h"
#include <wtf/MediaTime.h>

namespace WebCore {

struct GenericCueData {

    enum class Alignment : uint8_t { None, Start, Middle, End };
    enum class Status : uint8_t { Uninitialized, Partial, Complete };

    GenericCueData() = default;
    GenericCueData(Markable<InbandGenericCueIdentifier> uniqueId, const MediaTime& startTime, const MediaTime& endTime, const AtomString& id, const String& content, const String& fontName, double line, double position, double size, double baseFontSize, double relativeFontSize, const Color& foregroundColor, const Color& backgroundColor, const Color& highlightColor, GenericCueData::Alignment positionAlign, GenericCueData::Alignment align, GenericCueData::Status status)
        : m_uniqueId(uniqueId)
        , m_startTime(startTime)
        , m_endTime(endTime)
        , m_id(id)
        , m_content(content)
        , m_fontName(fontName)
        , m_line(line)
        , m_position(position)
        , m_size(size)
        , m_baseFontSize(baseFontSize)
        , m_relativeFontSize(relativeFontSize)
        , m_foregroundColor(foregroundColor)
        , m_backgroundColor(backgroundColor)
        , m_highlightColor(highlightColor)
        , m_positionAlign(positionAlign)
        , m_align(align)
        , m_status(status)
    {
        ASSERT(isValid());
    }

    bool isValid() const { return !!m_uniqueId; }
    bool equalNotConsideringTimesOrId(const GenericCueData&) const;

    Markable<InbandGenericCueIdentifier> m_uniqueId;
    MediaTime m_startTime;
    MediaTime m_endTime;
    AtomString m_id;
    String m_content;
    String m_fontName;
    double m_line { -1 };
    double m_position { -1 };
    double m_size { -1 };
    double m_baseFontSize { 0 };
    double m_relativeFontSize { 0 };
    Color m_foregroundColor;
    Color m_backgroundColor;
    Color m_highlightColor;
    Alignment m_positionAlign { Alignment::None };
    Alignment m_align { Alignment::None };
    Status m_status { Status::Uninitialized };
};

class InbandGenericCue : public RefCounted<InbandGenericCue> {
public:
    static Ref<InbandGenericCue> create() { return adoptRef(*new InbandGenericCue); }
    static Ref<InbandGenericCue> create(GenericCueData&& cueData) { return adoptRef(*new InbandGenericCue(WTFMove(cueData))); }

    InbandGenericCueIdentifier uniqueId() const { return *m_cueData.m_uniqueId; }

    MediaTime startTime() const { return m_cueData.m_startTime; }
    void setStartTime(const MediaTime& startTime) { m_cueData.m_startTime = startTime; }

    MediaTime endTime() const { return m_cueData.m_endTime; }
    void setEndTime(const MediaTime& endTime) { m_cueData.m_endTime = endTime; }

    const AtomString& id() const { return m_cueData.m_id; }
    void setId(const AtomString& id) { m_cueData.m_id = id; }

    const String& content() const { return m_cueData.m_content; }
    void setContent(const String& content) { m_cueData.m_content = content; }

    double line() const { return m_cueData.m_line; }
    void setLine(double line) { m_cueData.m_line = line; }

    double position() const { return m_cueData.m_position; }
    void setPosition(double position) { m_cueData.m_position = position; }

    GenericCueData::Alignment positionAlign() const { return m_cueData.m_positionAlign; }
    void setPositionAlign(GenericCueData::Alignment align) { m_cueData.m_positionAlign = align; }

    double size() const { return m_cueData.m_size; }
    void setSize(double size) { m_cueData.m_size = size; }

    GenericCueData::Alignment align() const { return m_cueData.m_align; }
    void setAlign(GenericCueData::Alignment align) { m_cueData.m_align = align; }

    const String& fontName() const { return m_cueData.m_fontName; }
    void setFontName(const String& fontName) { m_cueData.m_fontName = fontName; }

    double baseFontSize() const { return m_cueData.m_baseFontSize; }
    void setBaseFontSize(double baseFontSize) { m_cueData.m_baseFontSize = baseFontSize; }

    double relativeFontSize() const { return m_cueData.m_relativeFontSize; }
    void setRelativeFontSize(double relativeFontSize) { m_cueData.m_relativeFontSize = relativeFontSize; }

    const Color& foregroundColor() const { return m_cueData.m_foregroundColor; }
    void setForegroundColor(const Color& color) { m_cueData.m_foregroundColor = color; }

    const Color& backgroundColor() const { return m_cueData.m_backgroundColor; }
    void setBackgroundColor(const Color& color) { m_cueData.m_backgroundColor = color; }

    const Color& highlightColor() const { return m_cueData.m_highlightColor; }
    void setHighlightColor(const Color& color) { m_cueData.m_highlightColor = color; }

    GenericCueData::Status status() { return m_cueData.m_status; }
    void setStatus(GenericCueData::Status status) { m_cueData.m_status = status; }

    bool doesExtendCueData(const InbandGenericCue& other) const { return m_cueData.equalNotConsideringTimesOrId(other.m_cueData); }

    String toJSONString() const;

    const GenericCueData& cueData() const { return m_cueData; }

private:
    InbandGenericCue();
    explicit InbandGenericCue(GenericCueData&& cueData)
        : m_cueData(WTFMove(cueData))
    {
        ASSERT(m_cueData.isValid());
    }

    GenericCueData m_cueData;
};

} // namespace WebCore

namespace WTF {

template<typename Type> struct LogArgument;
template <>
struct LogArgument<WebCore::InbandGenericCue> {
    static String toString(const WebCore::InbandGenericCue& cue)
    {
        return cue.toJSONString();
    }
};

} // namespace WTF

#endif
