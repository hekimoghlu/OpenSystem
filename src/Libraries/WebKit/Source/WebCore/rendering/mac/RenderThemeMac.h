/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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

#if PLATFORM(MAC)

#import "RenderThemeCocoa.h"

OBJC_CLASS WebCoreRenderThemeNotificationObserver;

namespace WebCore {

class RenderStyle;

struct AttachmentLayout;

class RenderThemeMac final : public RenderThemeCocoa {
public:
    friend NeverDestroyed<RenderThemeMac>;

    // A method asking if the control changes its tint when the window has focus or not.
    bool controlSupportsTints(const RenderObject&) const final;

    // A general method asking if any control tinting is supported at all.
    bool supportsControlTints() const final { return true; }

    void inflateRectForControlRenderer(const RenderObject&, FloatRect&) final;
    void adjustRepaintRect(const RenderBox&, FloatRect&) final;

    bool isControlStyled(const RenderStyle&, const RenderStyle& userAgentStyle) const final;

    bool supportsSelectionForegroundColors(OptionSet<StyleColorOptions>) const final;

    Color platformActiveSelectionBackgroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformActiveSelectionForegroundColor(OptionSet<StyleColorOptions>) const final;
    Color transformSelectionBackgroundColor(const Color&, OptionSet<StyleColorOptions>) const final;
    Color platformInactiveSelectionBackgroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformInactiveSelectionForegroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformActiveListBoxSelectionBackgroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformActiveListBoxSelectionForegroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformInactiveListBoxSelectionBackgroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformInactiveListBoxSelectionForegroundColor(OptionSet<StyleColorOptions>) const final;
    Color platformFocusRingColor(OptionSet<StyleColorOptions>) const final;
    Color platformTextSearchHighlightColor(OptionSet<StyleColorOptions>) const final;
    Color platformAnnotationHighlightColor(OptionSet<StyleColorOptions>) const final;
    Color platformDefaultButtonTextColor(OptionSet<StyleColorOptions>) const final;
    Color platformAutocorrectionReplacementMarkerColor(OptionSet<StyleColorOptions>) const final;

    ScrollbarWidth scrollbarWidthStyleForPart(StyleAppearance) final { return ScrollbarWidth::Thin; }

    int minimumMenuListSize(const RenderStyle&) const final;

    void adjustSliderThumbSize(RenderStyle&, const Element*) const final;

    IntSize sliderTickSize() const final;
    int sliderTickOffsetFromTrackCenter() const final;

    LengthBox popupInternalPaddingBox(const RenderStyle&) const final;
    PopupMenuStyle::Size popupMenuSize(const RenderStyle&, IntRect&) const final;

    bool popsMenuByArrowKeys() const final { return true; }

    FloatSize meterSizeForBounds(const RenderMeter&, const FloatRect&) const final;
    bool supportsMeter(StyleAppearance) const final;

    void createColorWellSwatchSubtree(HTMLElement&) final;
    void setColorWellSwatchBackground(HTMLElement&, Color) final;

    IntRect progressBarRectForBounds(const RenderProgress&, const IntRect&) const final;

    // Controls color values returned from platformFocusRingColor(). systemColor() will be used when false.
    bool usesTestModeFocusRingColor() const;

    WEBCORE_EXPORT static RetainPtr<NSImage> iconForAttachment(const String& fileName, const String& attachmentType, const String& title);

private:
    RenderThemeMac();

    bool canPaint(const PaintInfo&, const Settings&, StyleAppearance) const final;
    bool canCreateControlPartForRenderer(const RenderObject&) const final;
    bool canCreateControlPartForBorderOnly(const RenderObject&) const final;
    bool canCreateControlPartForDecorations(const RenderObject&) const final;

    int baselinePosition(const RenderBox&) const final;

    bool supportsLargeFormControls() const final;

    void adjustMenuListStyle(RenderStyle&, const Element*) const final;

    void adjustMenuListButtonStyle(RenderStyle&, const Element*) const final;

    void adjustSliderTrackStyle(RenderStyle&, const Element*) const final;

    void adjustSliderThumbStyle(RenderStyle&, const Element*) const final;

    void adjustSearchFieldStyle(RenderStyle&, const Element*) const final;

    void adjustSearchFieldCancelButtonStyle(RenderStyle&, const Element*) const final;

    void adjustSearchFieldDecorationPartStyle(RenderStyle&, const Element*) const final;

    void adjustSearchFieldResultsDecorationPartStyle(RenderStyle&, const Element*) const final;

    void adjustSearchFieldResultsButtonStyle(RenderStyle&, const Element*) const final;

    Seconds switchAnimationVisuallyOnDuration() const final { return 300_ms; }
    bool hasSwitchHapticFeedback(SwitchTrigger trigger) const final { return trigger == SwitchTrigger::PointerTracking; }

    void adjustListButtonStyle(RenderStyle&, const Element*) const final;
    
#if ENABLE(SERVICE_CONTROLS)
    void adjustImageControlsButtonStyle(RenderStyle&, const Element*) const final;
#endif

#if ENABLE(ATTACHMENT_ELEMENT)
    LayoutSize attachmentIntrinsicSize(const RenderAttachment&) const final;
    bool paintAttachment(const RenderObject&, const PaintInfo&, const IntRect&) final;
#endif

private:
    String fileListNameForWidth(const FileList*, const FontCascade&, int width, bool multipleFilesAllowed) const final;

    Color systemColor(CSSValueID, OptionSet<StyleColorOptions>) const final;

    bool searchFieldShouldAppearAsTextField(const RenderStyle&) const final;

    std::span<const IntSize, 4> menuListSizes() const;
    std::span<const IntSize, 4> searchFieldSizes() const;
    std::span<const IntSize, 4> cancelButtonSizes() const;
    std::span<const IntSize, 4> resultsButtonSizes() const;
    void setSearchFieldSize(RenderStyle&) const;

#if ENABLE(SERVICE_CONTROLS)
    IntSize imageControlsButtonSize() const final;
    bool isImageControlsButton(const Element&) const final;
#endif

    mutable RetainPtr<NSPopUpButtonCell> m_popupButton;

    RetainPtr<WebCoreRenderThemeNotificationObserver> m_notificationObserver;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
