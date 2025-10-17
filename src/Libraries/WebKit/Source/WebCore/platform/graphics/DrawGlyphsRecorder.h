/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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

#include "AffineTransform.h"
#include "Color.h"
#include "Gradient.h"
#include "GraphicsContext.h"
#include "Pattern.h"
#include "TextFlags.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UniqueRef.h>

#if USE(CORE_TEXT)
#include <CoreGraphics/CoreGraphics.h>
#include <CoreText/CoreText.h>
#include <pal/spi/cf/CoreTextSPI.h>
#include <pal/spi/cg/CoreGraphicsSPI.h>
#endif

namespace WebCore {

class FloatPoint;
class Font;
class GlyphBuffer;
class GraphicsContext;

class DrawGlyphsRecorder {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DrawGlyphsRecorder);
    WTF_MAKE_NONCOPYABLE(DrawGlyphsRecorder);
public:
    enum class DeriveFontFromContext : bool { No, Yes };
    explicit DrawGlyphsRecorder(GraphicsContext&, float scaleFactor = 1, DeriveFontFromContext = DeriveFontFromContext::No);

    void drawGlyphs(const Font&, std::span<const GlyphBufferGlyph>, std::span<const GlyphBufferAdvance>, const FloatPoint& anchorPoint, FontSmoothingMode);

#if USE(CORE_TEXT)
    void drawNativeText(CTFontRef, CGFloat fontSize, CTLineRef, CGRect lineRect);

    void recordBeginLayer(CGRenderingStateRef, CGGStateRef, CGRect);
    void recordEndLayer(CGRenderingStateRef, CGGStateRef);
    void recordDrawGlyphs(CGRenderingStateRef, CGGStateRef, const CGAffineTransform*, std::span<const CGGlyph>, std::span<const CGPoint> positions);
    void recordDrawImage(CGRenderingStateRef, CGGStateRef, CGRect, CGImageRef);
    void recordDrawPath(CGRenderingStateRef, CGGStateRef, CGPathDrawingMode, CGPathRef);
#endif

private:
#if USE(CORE_TEXT)
    UniqueRef<GraphicsContext> createInternalContext();
#endif

    void drawBySplittingIntoOTSVGAndNonOTSVGRuns(const Font&, std::span<const GlyphBufferGlyph>, std::span<const GlyphBufferAdvance>, const FloatPoint& anchorPoint, FontSmoothingMode);
    void drawOTSVGRun(const Font&, std::span<const GlyphBufferGlyph>, std::span<const GlyphBufferAdvance>, const FloatPoint& anchorPoint, FontSmoothingMode);
    void drawNonOTSVGRun(const Font&, std::span<const GlyphBufferGlyph>, std::span<const GlyphBufferAdvance>, const FloatPoint& anchorPoint, FontSmoothingMode);

    void populateInternalState(const GraphicsContextState&);
    void populateInternalContext(const GraphicsContextState&);
    void prepareInternalContext(const Font&, FontSmoothingMode);
    void recordInitialColors();
    void concludeInternalContext();

    void updateFillBrush(const SourceBrush&);
    void updateStrokeBrush(const SourceBrush&);
    void updateCTM(const AffineTransform&);
    enum class ShadowsIgnoreTransforms {
        Unspecified,
        Yes,
        No
    };
    void updateShadow(const std::optional<GraphicsDropShadow>&, ShadowsIgnoreTransforms);

#if USE(CORE_TEXT)
    void updateFillColor(CGColorRef);
    void updateStrokeColor(CGColorRef);
    void updateShadow(CGStyleRef);
#endif

    GraphicsContext& m_owner;

#if USE(CORE_TEXT)
    UniqueRef<GraphicsContext> m_internalContext;
#endif

    const Font* m_originalFont { nullptr };

    const DeriveFontFromContext m_deriveFontFromContext;
    FontSmoothingMode m_smoothingMode { FontSmoothingMode::AutoSmoothing };

    AffineTransform m_originalTextMatrix;

    struct State {
        SourceBrush fillBrush;
        SourceBrush strokeBrush;
        AffineTransform ctm;
        std::optional<GraphicsDropShadow> dropShadow;
        bool ignoreTransforms { false };
    };
    State m_originalState;

#if USE(CORE_TEXT)
    RetainPtr<CGColorRef> m_initialFillColor;
    RetainPtr<CGColorRef> m_initialStrokeColor;
#endif
};

} // namespace WebCore
