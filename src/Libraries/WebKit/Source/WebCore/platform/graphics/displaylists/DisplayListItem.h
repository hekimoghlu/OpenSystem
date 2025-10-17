/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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

#include "RenderingResourceIdentifier.h"
#include <variant>
#include <wtf/OptionSet.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class ControlFactory;
class GraphicsContext;

namespace DisplayList {

class ResourceHeap;

class ApplyDeviceScaleFactor;
class BeginTransparencyLayer;
class BeginTransparencyLayerWithCompositeMode;
class ClearRect;
class ClearDropShadow;
class Clip;
class ClipRoundedRect;
class ClipOut;
class ClipOutRoundedRect;
class ClipOutToPath;
class ClipPath;
class ClipToImageBuffer;
class ConcatenateCTM;
class DrawControlPart;
class DrawDotsForDocumentMarker;
class DrawEllipse;
class DrawFilteredImageBuffer;
class DrawFocusRingPath;
class DrawFocusRingRects;
class DrawGlyphs;
class DrawDecomposedGlyphs;
class DrawImageBuffer;
class DrawLine;
class DrawLinesForText;
class DrawNativeImage;
class DrawPath;
class DrawPattern;
class DrawRect;
class DrawSystemImage;
class EndTransparencyLayer;
class FillCompositedRect;
class FillEllipse;
class FillPathSegment;
class FillPath;
class FillRect;
class FillRectWithColor;
class FillRectWithGradient;
class FillRectWithGradientAndSpaceTransform;
class FillRectWithRoundedHole;
class FillRoundedRect;
class ResetClip;
class Restore;
class Rotate;
class Save;
class Scale;
class SetCTM;
class SetInlineFillColor;
class SetInlineStroke;
class SetLineCap;
class SetLineDash;
class SetLineJoin;
class SetMiterLimit;
class SetState;
class StrokeEllipse;
class StrokeLine;
class StrokePathSegment;
class StrokePath;
class StrokeRect;
class Translate;
#if ENABLE(INLINE_PATH_DATA)
class FillLine;
class FillArc;
class FillClosedArc;
class FillQuadCurve;
class FillBezierCurve;
class StrokeArc;
class StrokeClosedArc;
class StrokeQuadCurve;
class StrokeBezierCurve;
#endif
#if USE(CG)
class ApplyFillPattern;
class ApplyStrokePattern;
#endif
class BeginPage;
class EndPage;
class SetURLForRect;

using Item = std::variant
    < ApplyDeviceScaleFactor
    , BeginTransparencyLayer
    , BeginTransparencyLayerWithCompositeMode
    , ClearRect
    , ClearDropShadow
    , Clip
    , ClipRoundedRect
    , ClipOut
    , ClipOutRoundedRect
    , ClipOutToPath
    , ClipPath
    , ClipToImageBuffer
    , ConcatenateCTM
    , DrawControlPart
    , DrawDotsForDocumentMarker
    , DrawEllipse
    , DrawFilteredImageBuffer
    , DrawFocusRingPath
    , DrawFocusRingRects
    , DrawGlyphs
    , DrawDecomposedGlyphs
    , DrawImageBuffer
    , DrawLine
    , DrawLinesForText
    , DrawNativeImage
    , DrawPath
    , DrawPattern
    , DrawRect
    , DrawSystemImage
    , EndTransparencyLayer
    , FillCompositedRect
    , FillEllipse
    , FillPathSegment
    , FillPath
    , FillRect
    , FillRectWithColor
    , FillRectWithGradient
    , FillRectWithGradientAndSpaceTransform
    , FillRectWithRoundedHole
    , FillRoundedRect
    , ResetClip
    , Restore
    , Rotate
    , Save
    , Scale
    , SetCTM
    , SetInlineFillColor
    , SetInlineStroke
    , SetLineCap
    , SetLineDash
    , SetLineJoin
    , SetMiterLimit
    , SetState
    , StrokeEllipse
    , StrokeLine
    , StrokePathSegment
    , StrokePath
    , StrokeRect
    , Translate
#if ENABLE(INLINE_PATH_DATA)
    , FillLine
    , FillArc
    , FillClosedArc
    , FillQuadCurve
    , FillBezierCurve
    , StrokeArc
    , StrokeClosedArc
    , StrokeQuadCurve
    , StrokeBezierCurve
#endif
#if USE(CG)
    , ApplyFillPattern
    , ApplyStrokePattern
#endif
    , BeginPage
    , EndPage
    , SetURLForRect
>;

enum class StopReplayReason : uint8_t {
    ReplayedAllItems,
    MissingCachedResource,
    InvalidItemOrExtent,
    OutOfMemory
};

struct ApplyItemResult {
    std::optional<StopReplayReason> stopReason;
    std::optional<RenderingResourceIdentifier> resourceIdentifier;
};

enum class ReplayOption : uint8_t {
    FlushAcceleratedImagesAndWaitForCompletion = 1 << 0,
};

enum class AsTextFlag : uint8_t {
    IncludePlatformOperations      = 1 << 0,
    IncludeResourceIdentifiers     = 1 << 1,
};

bool isValid(const Item&);

ApplyItemResult applyItem(GraphicsContext&, const ResourceHeap&, ControlFactory&, const Item&, OptionSet<ReplayOption>);

bool shouldDumpItem(const Item&, OptionSet<AsTextFlag>);

WEBCORE_EXPORT void dumpItem(TextStream&, const Item&, OptionSet<AsTextFlag>);

WEBCORE_EXPORT TextStream& operator<<(TextStream&, const Item&);
WEBCORE_EXPORT TextStream& operator<<(TextStream&, StopReplayReason);

} // namespace DisplayList
} // namespace WebCore
