/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#include "VTTCue.h"

namespace WebCore {

// A "generic" cue is a non-WebVTT cue, so it is not positioned/sized with the WebVTT logic.
class TextTrackCueGeneric final : public VTTCue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(TextTrackCueGeneric, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static Ref<TextTrackCueGeneric> create(ScriptExecutionContext&, const MediaTime& start, const MediaTime& end, const String& content);

    ExceptionOr<void> setPosition(const LineAndPositionSetting&) final;

    bool useDefaultPosition() const { return m_useDefaultPosition; }

    double baseFontSizeRelativeToVideoHeight() const { return m_baseFontSizeRelativeToVideoHeight; }
    void setBaseFontSizeRelativeToVideoHeight(double);

    double fontSizeMultiplier() const { return m_fontSizeMultiplier; }
    void setFontSizeMultiplier(double);

    const String& fontName() const { return m_fontName; }
    void setFontName(const String& name) { m_fontName = name; }

    const Color& foregroundColor() const { return m_foregroundColor; }
    void setForegroundColor(const Color& color) { m_foregroundColor = color; }
    
    const Color& backgroundColor() const { return m_backgroundColor; }
    void setBackgroundColor(const Color& color) { m_backgroundColor = color; }
    
    const Color& highlightColor() const { return m_highlightColor; }
    void setHighlightColor(const Color& color) { m_highlightColor = color; }

private:
    TextTrackCueGeneric(Document&, const MediaTime& start, const MediaTime& end, const String&);

    bool isOrderedBefore(const TextTrackCue*) const final;
    bool isPositionedAbove(const TextTrackCue*) const final;

    RefPtr<VTTCueBox> createDisplayTree() final;

    bool cueContentsMatch(const TextTrackCue&) const final;

    CueType cueType() const final { return ConvertedToWebVTT; }

    void toJSON(JSON::Object&) const final;

    Color m_foregroundColor;
    Color m_backgroundColor;
    Color m_highlightColor;
    double m_baseFontSizeRelativeToVideoHeight { 0 };
    double m_fontSizeMultiplier { 0 };
    String m_fontName;
    bool m_useDefaultPosition { true };
};

} // namespace WebCore

namespace WTF {

template<> struct LogArgument<WebCore::TextTrackCueGeneric> : LogArgument<WebCore::TextTrackCue> { };

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::TextTrackCueGeneric)
static bool isType(const WebCore::TextTrackCue& cue) { return cue.cueType() == WebCore::TextTrackCue::ConvertedToWebVTT; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
