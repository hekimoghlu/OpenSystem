/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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

#include "FloatRect.h"
#include "Image.h"
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/Seconds.h>
#include <wtf/Vector.h>

namespace WebCore {

class GraphicsContext;
class LocalFrame;

struct SimpleRange;

constexpr float dropShadowBlurRadius = 2;
constexpr float rimShadowBlurRadius = 1;
constexpr Seconds bounceAnimationDuration = 0.12_s;
constexpr Seconds timeBeforeFadeStarts = bounceAnimationDuration + 0.2_s;
constexpr float midBounceScale = 1.25;

enum class TextIndicatorLifetime : uint8_t {
    // The TextIndicator should indicate the text until dismissed.
    Permanent,

    // The TextIndicator should briefly indicate the text and then automatically dismiss.
    Temporary
};

enum class TextIndicatorDismissalAnimation : uint8_t {
    None,
    FadeOut
};

// FIXME: Move PresentationTransition to TextIndicatorWindow, because it's about presentation.
enum class TextIndicatorPresentationTransition : uint8_t {
    None,

    // These animations drive themselves.
    Bounce,
    BounceAndCrossfade,

    // This animation needs to be driven manually via TextIndicatorWindow::setAnimationProgress.
    FadeIn,
};

// Make sure to keep these in sync with the ones in Internals.idl.
enum class TextIndicatorOption : uint16_t {
    // Use the styled text color instead of forcing black text (the default)
    RespectTextColor = 1 << 0,

    // Paint backgrounds, even if they're not part of the Range
    PaintBackgrounds = 1 << 1,

    // Don't restrict painting to the given Range
    PaintAllContent = 1 << 2,

    // Take two snapshots:
    //    - one including the selection highlight and ignoring other painting-related options
    //    - one respecting the other painting-related options
    IncludeSnapshotWithSelectionHighlight = 1 << 3,

    // Tightly fit the content instead of expanding to cover the bounds of the selection highlight
    TightlyFitContent = 1 << 4,

    // If there are any non-inline or replaced elements in the Range, indicate the bounding rect
    // of the range instead of the individual subrects, and don't restrict painting to the given Range
    UseBoundingRectAndPaintAllContentForComplexRanges = 1 << 5,

    // By default, TextIndicator removes any margin if the given Range matches the
    // selection Range. If this option is set, maintain the margin in any case.
    IncludeMarginIfRangeMatchesSelection = 1 << 6,

    // By default, TextIndicator clips the indicated rects to the visible content rect.
    // If this option is set, expand the clip rect outward so that slightly offscreen content will be included.
    ExpandClipBeyondVisibleRect = 1 << 7,

    // By default, TextIndicator clips the indicated rects to the visible content rect.
    // If this option is set, do not clip to the visible rect.
    DoNotClipToVisibleRect = 1 << 8,

    // Include an additional snapshot of everything in view, with the exception of nodes within the currently selected range.
    IncludeSnapshotOfAllVisibleContentWithoutSelection = 1 << 9,

    // By default, TextIndicator uses text rects to size the snapshot. Enabling this flag causes it to use the bounds of the
    // selection rects that would enclose the given Range instead.
    // Currently, this is only supported on iOS.
    UseSelectionRectForSizing = 1 << 10,

    // Compute a background color to use when rendering a platter around the content image, falling back to a default if the
    // content's background is too complex to be captured by a single color.
    ComputeEstimatedBackgroundColor = 1 << 11,

    // By default, TextIndicator does not consider the user-select property.
    // If this option is set, expand the range to include the highest `user-select: all` ancestor.
    UseUserSelectAllCommonAncestor = 1 << 12,

    // If this option is set, exclude all content that is replaced by a separate render pass, like images, media, etc.
    SkipReplacedContent = 1 << 13,

    // If this option is set, perform the snapshot with 3x as the base scale, rather than the device scale factor
    SnapshotContentAt3xBaseScale = 1 << 14,
};

struct TextIndicatorData {
    FloatRect selectionRectInRootViewCoordinates;
    FloatRect textBoundingRectInRootViewCoordinates;
    FloatRect contentImageWithoutSelectionRectInRootViewCoordinates;
    Vector<FloatRect> textRectsInBoundingRectCoordinates;
    float contentImageScaleFactor { 1 };
    RefPtr<Image> contentImageWithHighlight;
    RefPtr<Image> contentImageWithoutSelection;
    RefPtr<Image> contentImage;
    Color estimatedBackgroundColor;
    TextIndicatorPresentationTransition presentationTransition { TextIndicatorPresentationTransition::None };
    OptionSet<TextIndicatorOption> options;
};

class TextIndicator : public RefCounted<TextIndicator> {
public:
    // FIXME: These are fairly Mac-specific, and they don't really belong here.
    // But they're needed at TextIndicator creation time, so they can't go in TextIndicatorWindow.
    // Maybe they can live in some Theme code somewhere?
    constexpr static float defaultHorizontalMargin { 2 };
    constexpr static float defaultVerticalMargin { 1 };

    WEBCORE_EXPORT static Ref<TextIndicator> create(const TextIndicatorData&);
    WEBCORE_EXPORT static RefPtr<TextIndicator> createWithSelectionInFrame(LocalFrame&, OptionSet<TextIndicatorOption>, TextIndicatorPresentationTransition, FloatSize margin = FloatSize(defaultHorizontalMargin, defaultVerticalMargin));
    WEBCORE_EXPORT static RefPtr<TextIndicator> createWithRange(const SimpleRange&, OptionSet<TextIndicatorOption>, TextIndicatorPresentationTransition, FloatSize margin = FloatSize(defaultHorizontalMargin, defaultVerticalMargin));

    WEBCORE_EXPORT ~TextIndicator();

    FloatRect selectionRectInRootViewCoordinates() const { return m_data.selectionRectInRootViewCoordinates; }
    FloatRect textBoundingRectInRootViewCoordinates() const { return m_data.textBoundingRectInRootViewCoordinates; }
    const Vector<FloatRect>& textRectsInBoundingRectCoordinates() const { return m_data.textRectsInBoundingRectCoordinates; }
    float contentImageScaleFactor() const { return m_data.contentImageScaleFactor; }
    Image* contentImageWithHighlight() const { return m_data.contentImageWithHighlight.get(); }
    Image* contentImage() const { return m_data.contentImage.get(); }
    RefPtr<Image> protectedContentImage() const { return contentImage(); }

    TextIndicatorPresentationTransition presentationTransition() const { return m_data.presentationTransition; }
    void setPresentationTransition(TextIndicatorPresentationTransition transition) { m_data.presentationTransition = transition; }

    TextIndicatorData data() const { return m_data; }

private:
    TextIndicator(const TextIndicatorData&);

    TextIndicatorData m_data;
};

} // namespace WebKit
