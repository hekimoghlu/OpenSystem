/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

#include "RenderElement.h"
#include "RenderObjectInlines.h"

namespace WebCore {

inline Overflow RenderElement::effectiveOverflowBlockDirection() const { return writingMode().isHorizontal() ? effectiveOverflowY() : effectiveOverflowX(); }
inline Overflow RenderElement::effectiveOverflowInlineDirection() const { return writingMode().isHorizontal() ? effectiveOverflowX() : effectiveOverflowY(); }
inline bool RenderElement::hasBackdropFilter() const { return style().hasBackdropFilter(); }
inline bool RenderElement::hasBackground() const { return style().hasBackground(); }
inline bool RenderElement::hasBlendMode() const { return style().hasBlendMode(); }
inline bool RenderElement::hasClip() const { return isOutOfFlowPositioned() && style().hasClip(); }
inline bool RenderElement::hasClipOrNonVisibleOverflow() const { return hasClip() || hasNonVisibleOverflow(); }
inline bool RenderElement::hasClipPath() const { return style().clipPath(); }
inline bool RenderElement::hasFilter() const { return style().hasFilter(); }
inline bool RenderElement::hasHiddenBackface() const { return style().backfaceVisibility() == BackfaceVisibility::Hidden; }
inline bool RenderElement::hasMask() const { return style().hasMask(); }
inline bool RenderElement::hasOutline() const { return style().hasOutline() || hasOutlineAnnotation(); }
inline bool RenderElement::hasShapeOutside() const { return style().shapeOutside(); }
inline bool RenderElement::isTransparent() const { return style().hasOpacity(); }
inline float RenderElement::opacity() const { return style().opacity(); }
inline FloatRect RenderElement::transformReferenceBoxRect() const { return transformReferenceBoxRect(style()); }
inline FloatRect RenderElement::transformReferenceBoxRect(const RenderStyle& style) const { return referenceBoxRect(transformBoxToCSSBoxType(style.transformBox())); }

#if HAVE(CORE_MATERIAL)
inline bool RenderElement::hasAppleVisualEffect() const { return style().hasAppleVisualEffect(); }
inline bool RenderElement::hasAppleVisualEffectRequiringBackdropFilter() const { return style().hasAppleVisualEffectRequiringBackdropFilter(); }
#endif

inline bool RenderElement::canContainAbsolutelyPositionedObjects() const
{
    return isRenderView()
        || style().position() != PositionType::Static
        || (canEstablishContainingBlockWithTransform() && hasTransformRelatedProperty())
        || (hasBackdropFilter() && !isDocumentElementRenderer())
#if HAVE(CORE_MATERIAL)
        || (hasAppleVisualEffectRequiringBackdropFilter() && !isDocumentElementRenderer())
#endif
        || (isRenderBlock() && style().willChange() && style().willChange()->createsContainingBlockForAbsolutelyPositioned(isDocumentElementRenderer()))
        || isRenderOrLegacyRenderSVGForeignObject()
        || shouldApplyLayoutContainment()
        || shouldApplyPaintContainment();
}

inline bool RenderElement::canContainFixedPositionObjects() const
{
    return isRenderView()
        || (canEstablishContainingBlockWithTransform() && hasTransformRelatedProperty())
        || (hasBackdropFilter() && !isDocumentElementRenderer())
#if HAVE(CORE_MATERIAL)
        || (hasAppleVisualEffectRequiringBackdropFilter() && !isDocumentElementRenderer())
#endif
        || (isRenderBlock() && style().willChange() && style().willChange()->createsContainingBlockForOutOfFlowPositioned(isDocumentElementRenderer()))
        || isRenderOrLegacyRenderSVGForeignObject()
        || shouldApplyLayoutContainment()
        || shouldApplyPaintContainment();
}

inline bool RenderElement::createsGroupForStyle(const RenderStyle& style)
{
    return style.hasOpacity()
        || style.hasMask()
        || style.clipPath()
        || style.hasFilter()
        || style.hasBackdropFilter()
#if HAVE(CORE_MATERIAL)
        || style.hasAppleVisualEffect()
#endif
        || style.hasBlendMode();
}

inline bool RenderElement::shouldApplyAnyContainment() const
{
    return shouldApplyLayoutContainment() || shouldApplySizeContainment() || shouldApplyInlineSizeContainment() || shouldApplyStyleContainment() || shouldApplyPaintContainment();
}

inline bool RenderElement::shouldApplySizeOrInlineSizeContainment() const
{
    return shouldApplySizeContainment() || shouldApplyInlineSizeContainment();
}

inline bool RenderElement::shouldApplyLayoutContainment() const
{
    return element() && WebCore::shouldApplyLayoutContainment(style(), *element());
}

inline bool RenderElement::shouldApplySizeContainment() const
{
    return element() && WebCore::shouldApplySizeContainment(style(), *element());
}

inline bool RenderElement::shouldApplyInlineSizeContainment() const
{
    return element() && WebCore::shouldApplyInlineSizeContainment(style(), *element());
}

inline bool RenderElement::shouldApplyStyleContainment() const
{
    return element() && WebCore::shouldApplyStyleContainment(style(), *element());
}

inline bool RenderElement::shouldApplyPaintContainment() const
{
    return element() && WebCore::shouldApplyPaintContainment(style(), *element());
}

inline bool RenderElement::visibleToHitTesting(const std::optional<HitTestRequest>& request) const
{
    auto visibility = !request || request->userTriggered() ? style().usedVisibility() : style().visibility();
    return visibility == Visibility::Visible
        && !isSkippedContent()
        && ((request && request->ignoreCSSPointerEventsProperty()) || usedPointerEvents() != PointerEvents::None);
}

inline int adjustForAbsoluteZoom(int value, const RenderElement& renderer)
{
    return adjustForAbsoluteZoom(value, renderer.style());
}

inline LayoutSize adjustLayoutSizeForAbsoluteZoom(LayoutSize size, const RenderElement& renderer)
{
    return adjustLayoutSizeForAbsoluteZoom(size, renderer.style());
}

inline LayoutUnit adjustLayoutUnitForAbsoluteZoom(LayoutUnit value, const RenderElement& renderer)
{
    return adjustLayoutUnitForAbsoluteZoom(value, renderer.style());
}

inline bool isSkippedContentRoot(const RenderElement& renderer)
{
    return renderer.element() && WebCore::isSkippedContentRoot(renderer.style(), *renderer.element());
}

} // namespace WebCore
