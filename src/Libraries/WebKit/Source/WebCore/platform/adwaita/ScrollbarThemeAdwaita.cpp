/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#include "ScrollbarThemeAdwaita.h"

#if USE(THEME_ADWAITA)

#include "Color.h"
#include "FloatRoundedRect.h"
#include "GraphicsContext.h"
#include "PlatformMouseEvent.h"
#include "ScrollableArea.h"
#include "Scrollbar.h"
#include "ThemeAdwaita.h"

#if PLATFORM(GTK) || PLATFORM(WPE)
#include "SystemSettings.h"
#endif

#if PLATFORM(GTK)
#include "GtkUtilities.h"
#endif

namespace WebCore {

static const unsigned scrollbarSize = 21;
static const unsigned scrollbarBorderSize = 1;
static const unsigned thumbBorderSize = 1;
static const unsigned overlayThumbSize = 3;
static const unsigned minimumThumbSize = 40;
static const unsigned horizThumbMargin = 6;
static const unsigned horizOverlayThumbMargin = 3;
static const unsigned vertThumbMargin = 7;

static constexpr auto scrollbarBackgroundColorLight = Color::white;
static constexpr auto scrollbarBorderColorLight = Color::black.colorWithAlphaByte(38);
static constexpr auto overlayThumbBorderColorLight = Color::white.colorWithAlphaByte(102);
static constexpr auto overlayTroughColorLight = Color::black.colorWithAlphaByte(25);
static constexpr auto thumbHoveredColorLight = Color::black.colorWithAlphaByte(102);
static constexpr auto thumbPressedColorLight = Color::black.colorWithAlphaByte(153);
static constexpr auto thumbColorLight = Color::black.colorWithAlphaByte(51);

static constexpr auto scrollbarBackgroundColorDark = SRGBA<uint8_t> { 30, 30, 30 };
static constexpr auto scrollbarBorderColorDark = Color::white.colorWithAlphaByte(38);
static constexpr auto overlayThumbBorderColorDark = Color::black.colorWithAlphaByte(51);
static constexpr auto overlayTroughColorDark = Color::white.colorWithAlphaByte(25);
static constexpr auto thumbHoveredColorDark = Color::white.colorWithAlphaByte(102);
static constexpr auto thumbPressedColorDark = Color::white.colorWithAlphaByte(153);
static constexpr auto thumbColorDark = Color::white.colorWithAlphaByte(51);

void ScrollbarThemeAdwaita::updateScrollbarOverlayStyle(Scrollbar& scrollbar)
{
    scrollbar.invalidate();
}

bool ScrollbarThemeAdwaita::usesOverlayScrollbars() const
{
#if PLATFORM(GTK) && !USE(GTK4)
    if (!g_strcmp0(g_getenv("GTK_OVERLAY_SCROLLING"), "0"))
        return false;
#endif
#if PLATFORM(GTK) || PLATFORM(WPE)
    return SystemSettings::singleton().overlayScrolling().value_or(true);
#else
    return true;
#endif
}

int ScrollbarThemeAdwaita::scrollbarThickness(ScrollbarWidth scrollbarWidth, ScrollbarExpansionState, OverlayScrollbarSizeRelevancy overlayRelevancy)
{
    if (scrollbarWidth == ScrollbarWidth::None || (usesOverlayScrollbars() && overlayRelevancy == OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize))
        return 0;
    return scrollbarSize;
}

int ScrollbarThemeAdwaita::minimumThumbLength(Scrollbar&)
{
    return minimumThumbSize;
}

bool ScrollbarThemeAdwaita::hasButtons(Scrollbar&)
{
    return false;
}

bool ScrollbarThemeAdwaita::hasThumb(Scrollbar& scrollbar)
{
    return thumbLength(scrollbar) > 0;
}

IntRect ScrollbarThemeAdwaita::backButtonRect(Scrollbar&, ScrollbarPart, bool)
{
    return { };
}

IntRect ScrollbarThemeAdwaita::forwardButtonRect(Scrollbar&, ScrollbarPart, bool)
{
    return { };
}

IntRect ScrollbarThemeAdwaita::trackRect(Scrollbar& scrollbar, bool)
{
    return scrollbar.frameRect();
}

bool ScrollbarThemeAdwaita::paint(Scrollbar& scrollbar, GraphicsContext& graphicsContext, const IntRect& damageRect)
{
    if (graphicsContext.paintingDisabled())
        return false;

    if (!scrollbar.enabled() && usesOverlayScrollbars())
        return true;

    IntRect rect = scrollbar.frameRect();
    if (!rect.intersects(damageRect))
        return true;

    double opacity;
    if (usesOverlayScrollbars())
        opacity = scrollbar.opacity();
    else
        opacity = 1;
    if (!opacity)
        return true;

    SRGBA<uint8_t> scrollbarBackgroundColor;
    SRGBA<uint8_t> scrollbarBorderColor;
    SRGBA<uint8_t> overlayThumbBorderColor;
    SRGBA<uint8_t> overlayTroughBorderColor;
    SRGBA<uint8_t> overlayTroughColor;
    SRGBA<uint8_t> thumbHoveredColor;
    SRGBA<uint8_t> thumbPressedColor;
    SRGBA<uint8_t> thumbColor;

    if (scrollbar.scrollableArea().useDarkAppearanceForScrollbars()) {
        scrollbarBackgroundColor = scrollbarBackgroundColorDark;
        scrollbarBorderColor = scrollbarBorderColorDark;
        overlayThumbBorderColor = overlayThumbBorderColorDark;
        overlayTroughBorderColor = overlayThumbBorderColorDark;
        overlayTroughColor = overlayTroughColorDark;
        thumbHoveredColor = thumbHoveredColorDark;
        thumbPressedColor = thumbPressedColorDark;
        thumbColor = thumbColorDark;
    } else {
        scrollbarBackgroundColor = scrollbarBackgroundColorLight;
        scrollbarBorderColor = scrollbarBorderColorLight;
        overlayThumbBorderColor = overlayThumbBorderColorLight;
        overlayTroughBorderColor = overlayThumbBorderColorLight;
        overlayTroughColor = overlayTroughColorLight;
        thumbHoveredColor = thumbHoveredColorLight;
        thumbPressedColor = thumbPressedColorLight;
        thumbColor = thumbColorLight;
    }

    GraphicsContextStateSaver stateSaver(graphicsContext);
    if (opacity != 1) {
        graphicsContext.clip(damageRect);
        graphicsContext.beginTransparencyLayer(opacity);
    }

    int thumbSize = scrollbarSize - scrollbarBorderSize - horizThumbMargin * 2;

    if (!usesOverlayScrollbars()) {
        graphicsContext.fillRect(rect, scrollbarBackgroundColor);

        IntRect frame = rect;
        if (scrollbar.orientation() == ScrollbarOrientation::Vertical) {
            if (scrollbar.scrollableArea().shouldPlaceVerticalScrollbarOnLeft())
                frame.move(frame.width() - scrollbarBorderSize, 0);
            frame.setWidth(scrollbarBorderSize);
        } else
            frame.setHeight(scrollbarBorderSize);
        graphicsContext.fillRect(frame, scrollbarBorderColor);
    } else if (scrollbar.hoveredPart() != NoPart) {
        int thumbCornerSize = thumbSize / 2;
        FloatSize corner(thumbCornerSize, thumbCornerSize);
        FloatSize borderCorner(thumbCornerSize + thumbBorderSize, thumbCornerSize + thumbBorderSize);

        IntRect trough = rect;
        if (scrollbar.orientation() == ScrollbarOrientation::Vertical) {
            if (scrollbar.scrollableArea().shouldPlaceVerticalScrollbarOnLeft())
                trough.move(scrollbarSize - (scrollbarSize / 2 + thumbSize / 2) - scrollbarBorderSize, vertThumbMargin);
            else
                trough.move(scrollbarSize - (scrollbarSize / 2 + thumbSize / 2), vertThumbMargin);
            trough.setWidth(thumbSize);
            trough.setHeight(rect.height() - vertThumbMargin * 2);
        } else {
            trough.move(vertThumbMargin, scrollbarSize - (scrollbarSize / 2 + thumbSize / 2));
            trough.setWidth(rect.width() - vertThumbMargin * 2);
            trough.setHeight(thumbSize);
        }

        IntRect troughBorder(trough);
        troughBorder.inflate(thumbBorderSize);

        Path path;
        path.addRoundedRect(trough, corner);
        graphicsContext.setFillRule(WindRule::NonZero);
        graphicsContext.setFillColor(overlayTroughColor);
        graphicsContext.fillPath(path);
        path.clear();

        path.addRoundedRect(trough, corner);
        path.addRoundedRect(troughBorder, borderCorner);
        graphicsContext.setFillRule(WindRule::EvenOdd);
        graphicsContext.setFillColor(overlayTroughBorderColor);
        graphicsContext.fillPath(path);
    }

    int thumbCornerSize;
    int thumbPos = thumbPosition(scrollbar);
    int thumbLen = thumbLength(scrollbar);
    IntRect thumb = rect;
    if (scrollbar.hoveredPart() == NoPart && usesOverlayScrollbars()) {
        thumbCornerSize = overlayThumbSize / 2;

        if (scrollbar.orientation() == ScrollbarOrientation::Vertical) {
            if (scrollbar.scrollableArea().shouldPlaceVerticalScrollbarOnLeft())
                thumb.move(horizOverlayThumbMargin, thumbPos + vertThumbMargin);
            else
                thumb.move(scrollbarSize - overlayThumbSize - horizOverlayThumbMargin, thumbPos + vertThumbMargin);
            thumb.setWidth(overlayThumbSize);
            thumb.setHeight(thumbLen - vertThumbMargin * 2);
        } else {
            thumb.move(thumbPos + vertThumbMargin, scrollbarSize - overlayThumbSize - horizOverlayThumbMargin);
            thumb.setWidth(thumbLen - vertThumbMargin * 2);
            thumb.setHeight(overlayThumbSize);
        }
    } else {
        thumbCornerSize = thumbSize / 2;

        if (scrollbar.orientation() == ScrollbarOrientation::Vertical) {
            if (scrollbar.scrollableArea().shouldPlaceVerticalScrollbarOnLeft())
                thumb.move(scrollbarSize - (scrollbarSize / 2 + thumbSize / 2) - scrollbarBorderSize, thumbPos + vertThumbMargin);
            else
                thumb.move(scrollbarSize - (scrollbarSize / 2 + thumbSize / 2), thumbPos + vertThumbMargin);
            thumb.setWidth(thumbSize);
            thumb.setHeight(thumbLen - vertThumbMargin * 2);
        } else {
            thumb.move(thumbPos + vertThumbMargin, scrollbarSize - (scrollbarSize / 2 + thumbSize / 2));
            thumb.setWidth(thumbLen - vertThumbMargin * 2);
            thumb.setHeight(thumbSize);
        }
    }

    FloatSize corner(thumbCornerSize, thumbCornerSize);
    FloatSize borderCorner(thumbCornerSize + thumbBorderSize, thumbCornerSize + thumbBorderSize);

    Path path;

    path.addRoundedRect(thumb, corner);
    graphicsContext.setFillRule(WindRule::NonZero);
    if (scrollbar.pressedPart() == ThumbPart)
        graphicsContext.setFillColor(thumbPressedColor);
    else if (scrollbar.hoveredPart() == ThumbPart)
        graphicsContext.setFillColor(thumbHoveredColor);
    else
        graphicsContext.setFillColor(thumbColor);
    graphicsContext.fillPath(path);
    path.clear();

    if (usesOverlayScrollbars()) {
        IntRect thumbBorder(thumb);
        thumbBorder.inflate(thumbBorderSize);

        path.addRoundedRect(thumb, corner);
        path.addRoundedRect(thumbBorder, borderCorner);
        graphicsContext.setFillRule(WindRule::EvenOdd);
        graphicsContext.setFillColor(overlayThumbBorderColor);
        graphicsContext.fillPath(path);
    }

    if (opacity != 1)
        graphicsContext.endTransparencyLayer();

    return true;
}

void ScrollbarThemeAdwaita::paintScrollCorner(ScrollableArea& scrollableArea, GraphicsContext& graphicsContext, const IntRect& cornerRect)
{
    if (graphicsContext.paintingDisabled())
        return;

    SRGBA<uint8_t> scrollbarBackgroundColor;
    SRGBA<uint8_t> scrollbarBorderColor;

    if (scrollableArea.useDarkAppearanceForScrollbars()) {
        scrollbarBackgroundColor = scrollbarBackgroundColorDark;
        scrollbarBorderColor = scrollbarBorderColorDark;
    } else {
        scrollbarBackgroundColor = scrollbarBackgroundColorLight;
        scrollbarBorderColor = scrollbarBorderColorLight;
    }

    IntRect borderRect = IntRect(cornerRect.location(), IntSize(scrollbarBorderSize, scrollbarBorderSize));

    if (scrollableArea.shouldPlaceVerticalScrollbarOnLeft())
        borderRect.move(cornerRect.width() - scrollbarBorderSize, 0);

    graphicsContext.fillRect(cornerRect, scrollbarBackgroundColor);
    graphicsContext.fillRect(borderRect, scrollbarBorderColor);
}

ScrollbarButtonPressAction ScrollbarThemeAdwaita::handleMousePressEvent(Scrollbar&, const PlatformMouseEvent& event, ScrollbarPart pressedPart)
{
    bool warpSlider = false;
    switch (pressedPart) {
    case BackTrackPart:
    case ForwardTrackPart:
#if PLATFORM(GTK) || PLATFORM(WPE)
        warpSlider = SystemSettings::singleton().primaryButtonWarpsSlider().value_or(true);
#endif
        // The shift key or middle/right button reverses the sense.
        if (event.shiftKey() || event.button() != MouseButton::Left)
            warpSlider = !warpSlider;
        return warpSlider ?
            ScrollbarButtonPressAction::CenterOnThumb:
            ScrollbarButtonPressAction::Scroll;
    case ThumbPart:
        if (event.button() != MouseButton::Right)
            return ScrollbarButtonPressAction::StartDrag;
        break;
    case BackButtonStartPart:
    case ForwardButtonStartPart:
    case BackButtonEndPart:
    case ForwardButtonEndPart:
        return ScrollbarButtonPressAction::Scroll;
    default:
        break;
    }

    return ScrollbarButtonPressAction::None;
}

#if !PLATFORM(GTK) || USE(GTK4) || USE(SKIA)
ScrollbarTheme& ScrollbarTheme::nativeTheme()
{
    static ScrollbarThemeAdwaita theme;
    return theme;
}
#endif

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
