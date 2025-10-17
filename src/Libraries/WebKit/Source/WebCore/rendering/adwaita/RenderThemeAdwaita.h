/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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

#if USE(THEME_ADWAITA)

#include "RenderTheme.h"

namespace WebCore {

class RenderThemeAdwaita : public RenderTheme {
public:
    virtual ~RenderThemeAdwaita();

    bool canCreateControlPartForRenderer(const RenderObject&) const final;
    bool canCreateControlPartForBorderOnly(const RenderObject&) const final;
    bool canCreateControlPartForDecorations(const RenderObject&) const final;

    bool isControlStyled(const RenderStyle&, const RenderStyle& userAgentStyle) const final;

    void setAccentColor(const Color&);

private:
    String extraDefaultStyleSheet() final;
#if ENABLE(VIDEO)
    Vector<String, 2> mediaControlsScripts() final;
    String mediaControlsStyleSheet() final;
#endif

#if ENABLE(VIDEO)
    String mediaControlsBase64StringForIconNameAndType(const String&, const String&) final;
    String mediaControlsFormattedStringForDuration(double) final;

    String m_mediaControlsStyleSheet;
#endif // ENABLE(VIDEO)

    bool supportsHover() const final { return true; }
    bool supportsFocusRing(const RenderStyle&) const final;
    bool supportsSelectionForegroundColors(OptionSet<StyleColorOptions>) const final { return false; }
    bool supportsListBoxSelectionForegroundColors(OptionSet<StyleColorOptions>) const final { return true; }
    bool shouldHaveCapsLockIndicator(const HTMLInputElement&) const final;

    Color platformActiveSelectionBackgroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformInactiveSelectionBackgroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformActiveSelectionForegroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformInactiveSelectionForegroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformActiveListBoxSelectionBackgroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformActiveListBoxSelectionForegroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformInactiveListBoxSelectionBackgroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformInactiveListBoxSelectionForegroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformFocusRingColor(OptionSet<StyleColorOptions>) const final;
    void platformColorsDidChange() final;

    void adjustTextFieldStyle(RenderStyle&, const Element*) const final;
    void adjustTextAreaStyle(RenderStyle&, const Element*) const final;
    void adjustSearchFieldStyle(RenderStyle&, const Element*) const final;

    bool popsMenuBySpaceOrReturn() const final { return true; }
    void adjustMenuListStyle(RenderStyle&, const Element*) const override;
    void adjustMenuListButtonStyle(RenderStyle&, const Element*) const override;
    LengthBox popupInternalPaddingBox(const RenderStyle&) const final;

    Seconds animationRepeatIntervalForProgressBar(const RenderProgress&) const final;
    IntRect progressBarRectForBounds(const RenderProgress&, const IntRect&) const final;

    void adjustSliderThumbSize(RenderStyle&, const Element*) const final;

    Color systemColor(CSSValueID, OptionSet<StyleColorOptions>) const final;

    IntSize sliderTickSize() const final;
    int sliderTickOffsetFromTrackCenter() const final;
    void adjustListButtonStyle(RenderStyle&, const Element*) const final;

#if PLATFORM(GTK) || PLATFORM(WPE)
    std::optional<Seconds> caretBlinkInterval() const override;
#endif
};

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
