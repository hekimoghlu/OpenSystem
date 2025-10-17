/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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

#include "HTMLElement.h"
#include "SpeechSynthesisUtterance.h"
#include "TextTrackCue.h"
#include "VTTRegion.h"
#include <wtf/LoggerHelper.h>
#include <wtf/TypeCasts.h>

namespace WebCore {

class DocumentFragment;
class HTMLDivElement;
class HTMLSpanElement;
class VTTCue;
class VTTScanner;
class WebVTTCueData;

enum class VTTDirectionSetting : uint8_t {
    Horizontal,
    VerticalGrowingLeft,
    VerticalGrowingRight,

    // IDL equivalents:
    EmptyString = Horizontal,
    Rl = VerticalGrowingLeft,
    Lr = VerticalGrowingRight,

    // For static-assert convenience.
    MaxValue = VerticalGrowingRight,
};

enum class VTTLineAlignSetting : uint8_t {
    Start,
    Center,
    End,
};

enum class VTTPositionAlignSetting : uint8_t {
    LineLeft,
    Center,
    LineRight,
    Auto,
};

enum class VTTAlignSetting : uint8_t {
    Start,
    Center,
    End,
    Left,
    Right,

    // For static-assert convenience.
    MaxValue = Right,
};

// ----------------------------

class VTTCueBox : public TextTrackCueBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(VTTCueBox);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(VTTCueBox);
public:
    static Ref<VTTCueBox> create(Document&, VTTCue&);

    void applyCSSProperties() override;
    void applyCSSPropertiesWithRegion();

protected:
    VTTCueBox(Document&, VTTCue&);

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

private:
    WeakPtr<VTTCue> m_cue;
};

// ----------------------------

class VTTCue
    : public TextTrackCue
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(VTTCue);
public:
    static Ref<VTTCue> create(Document&, double start, double end, String&& content);
    static Ref<VTTCue> create(Document&, const WebVTTCueData&);

    virtual ~VTTCue();

    enum AutoKeyword { Auto };
    using LineAndPositionSetting = std::variant<double, AutoKeyword>;

    using DirectionSetting = VTTDirectionSetting;
    static constexpr size_t DirectionSettingCount = static_cast<size_t>(DirectionSetting::VerticalGrowingRight) + 1;

    using LineAlignSetting = VTTLineAlignSetting;
    static constexpr size_t LineAlignSettingCount = static_cast<size_t>(LineAlignSetting::End) + 1;

    using PositionAlignSetting = VTTPositionAlignSetting;
    using AlignSetting = VTTAlignSetting;

    void setTrack(TextTrack*);

    DirectionSetting vertical() const { return m_writingDirection; }
    void setVertical(DirectionSetting);

    bool snapToLines() const { return m_snapToLines; }
    void setSnapToLines(bool);

    LineAndPositionSetting line() const;
    void setLine(const LineAndPositionSetting&);

    LineAlignSetting lineAlign() const { return m_lineAlignment; }
    void setLineAlign(LineAlignSetting);

    LineAndPositionSetting position() const;
    virtual ExceptionOr<void> setPosition(const LineAndPositionSetting&);

    PositionAlignSetting positionAlign() const { return m_positionAlignment; }
    void setPositionAlign(PositionAlignSetting);

    double size() const { return m_cueSize; }
    ExceptionOr<void> setSize(double);

    AlignSetting align() const { return m_cueAlignment; }
    void setAlign(AlignSetting);

    const String& text() const final { return m_content; }
    void setText(const String&);

    const String& cueSettings() const { return m_settings; }
    void setCueSettings(const String&);

    RefPtr<DocumentFragment> getCueAsHTML() final;
    RefPtr<DocumentFragment> createCueRenderingTree();

    void notifyRegionWhenRemovingDisplayTree(bool);

    VTTRegion* region();
    void setRegion(VTTRegion*);

    const String& regionId();

    void setIsActive(bool) override;

    bool hasDisplayTree() const { return m_displayTree; }
    RefPtr<TextTrackCueBox> getDisplayTree() final;
    HTMLSpanElement& element() const { return m_cueHighlightBox; }
    HTMLDivElement& backdrop() const { return m_cueBackdropBox; }

    void updateDisplayTree(const MediaTime&) final;
    void removeDisplayTree() final;
    void markFutureAndPastNodes(ContainerNode*, const MediaTime&, const MediaTime&);

    int calculateComputedLinePosition() const;
    std::pair<double, double> getPositionCoordinates() const;

    using DisplayPosition = std::pair<std::optional<double>, std::optional<double>>;
    const DisplayPosition& getCSSPosition() const { return m_displayPosition; };

    CSSValueID getCSSAlignment() const;
    int getCSSSize() const;
    CSSValueID getCSSWritingDirection() const;
    CSSValueID getCSSWritingMode() const;

    void recalculateStyles() final { m_displayTreeShouldChange = true; }
    void setFontSize(int, bool important) override;
    int fontSize() const { return m_fontSize; }
    bool fontSizeIsImportant() const { return m_fontSizeIsImportant; }

    CueType cueType() const override { return WebVTT; }
    bool isRenderable() const final { return !m_content.isEmpty(); }

    void didChange(bool = false) final;

    double calculateComputedTextPosition() const;
    PositionAlignSetting calculateComputedPositionAlignment() const;
    double calculateMaximumSize() const;

#if ENABLE(SPEECH_SYNTHESIS)
    RefPtr<SpeechSynthesisUtterance> speechUtterance() const { return m_speechUtterance; }
#endif

    const LineAndPositionSetting& left() const { return m_left; }
    const LineAndPositionSetting& top() const { return m_top; }
    const LineAndPositionSetting& width() const { return m_width; }
    const LineAndPositionSetting& height() const { return m_height; }

protected:
    VTTCue(Document&, const MediaTime& start, const MediaTime& end, String&& content);

    bool cueContentsMatch(const TextTrackCue&) const override;

    virtual RefPtr<VTTCueBox> createDisplayTree();
    VTTCueBox* displayTreeInternal();

    void toJSON(JSON::Object&) const override;

private:
    VTTCue(Document&, const WebVTTCueData&);

    void createWebVTTNodeTree();

    void parseSettings(const String&);

    void determineTextDirection();
    void calculateDisplayParameters();
    void calculateDisplayParametersWithRegion();
    void obtainCSSBoxes();

    enum CueSetting {
        None,
        Vertical,
        Line,
        Position,
        Size,
        Align,
        Region
    };
    CueSetting settingName(VTTScanner&);

    void prepareToSpeak(SpeechSynthesis&, double, double, SpeakCueCompletionHandler&&) final;
    void beginSpeaking() final;
    void pauseSpeaking() final;
    void cancelSpeaking() final;

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return *m_logger; }
    uint64_t logIdentifier() const final;
    WTFLogChannel& logChannel() const final;
    ASCIILiteral logClassName() const final { return "VTTCue"_s; }
#endif

    String m_content;
    String m_settings;
    std::optional<double> m_linePosition;
    std::optional<double> m_computedLinePosition;
    std::optional<double> m_textPosition;
    double m_cueSize { 100 };

    DirectionSetting m_writingDirection { DirectionSetting::Horizontal };
    AlignSetting m_cueAlignment { AlignSetting::Center };

    RefPtr<VTTRegion> m_region;
    String m_parsedRegionId;

    RefPtr<DocumentFragment> m_webVTTNodeTree;
    Ref<HTMLSpanElement> m_cueHighlightBox;
    Ref<HTMLDivElement> m_cueBackdropBox;
    RefPtr<VTTCueBox> m_displayTree;
#if ENABLE(SPEECH_SYNTHESIS)
    RefPtr<SpeechSynthesis> m_speechSynthesis;
    RefPtr<SpeechSynthesisUtterance> m_speechUtterance;
#endif

    CSSValueID m_displayDirection { CSSValueLtr };
    double m_displaySize { 0 };
    DisplayPosition m_displayPosition;

    MediaTime m_originalStartTime;

    int m_fontSize { 0 };
    bool m_fontSizeIsImportant { false };

    bool m_snapToLines : 1;
    bool m_displayTreeShouldChange : 1;
    bool m_notifyRegion : 1;

    PositionAlignSetting m_positionAlignment { PositionAlignSetting::Auto };
    LineAlignSetting m_lineAlignment { LineAlignSetting::Start };

    LineAndPositionSetting m_left { Auto };
    LineAndPositionSetting m_top { Auto };
    LineAndPositionSetting m_width { Auto };
    LineAndPositionSetting m_height { Auto };

#if !RELEASE_LOG_DISABLED
    mutable RefPtr<Logger> m_logger;
    mutable uint64_t m_logIdentifier { 0 };
#endif
};

} // namespace WebCore

namespace WTF {

template<> struct LogArgument<WebCore::VTTCue> : LogArgument<WebCore::TextTrackCue> { };

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::VTTCue)
static bool isType(const WebCore::TextTrackCue& cue) { return cue.cueType() == WebCore::TextTrackCue::WebVTT || cue.cueType() == WebCore::TextTrackCue::ConvertedToWebVTT; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
