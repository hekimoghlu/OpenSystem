/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#include "LegacyRenderSVGResourcePattern.h"

#include "ElementChildIteratorInlines.h"
#include "GraphicsContext.h"
#include "LegacyRenderSVGRoot.h"
#include "LocalFrameView.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFitToViewBox.h"
#include "SVGRenderStyle.h"
#include "SVGRenderingContext.h"
#include "SVGResources.h"
#include "SVGResourcesCache.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGResourcePattern);

LegacyRenderSVGResourcePattern::LegacyRenderSVGResourcePattern(SVGPatternElement& element, RenderStyle&& style)
    : LegacyRenderSVGResourceContainer(Type::LegacySVGResourcePattern, element, WTFMove(style))
{
}

LegacyRenderSVGResourcePattern::~LegacyRenderSVGResourcePattern() = default;

SVGPatternElement& LegacyRenderSVGResourcePattern::patternElement() const
{
    return downcast<SVGPatternElement>(LegacyRenderSVGResourceContainer::element());
}

Ref<SVGPatternElement> LegacyRenderSVGResourcePattern::protectedPatternElement() const
{
    return patternElement();
}

void LegacyRenderSVGResourcePattern::removeAllClientsFromCache()
{
    m_patternMap.clear();
    m_shouldCollectPatternAttributes = true;
}

void LegacyRenderSVGResourcePattern::removeAllClientsFromCacheAndMarkForInvalidationIfNeeded(bool markForInvalidation, SingleThreadWeakHashSet<RenderObject>* visitedRenderers)
{
    removeAllClientsFromCache();
    markAllClientsForInvalidationIfNeeded(markForInvalidation ? RepaintInvalidation : ParentOnlyInvalidation, visitedRenderers);
}

void LegacyRenderSVGResourcePattern::removeClientFromCache(RenderElement& client)
{
    m_patternMap.remove(client);
}

void LegacyRenderSVGResourcePattern::collectPatternAttributes(PatternAttributes& attributes) const
{
    const LegacyRenderSVGResourcePattern* current = this;

    while (current) {
        Ref pattern = current->patternElement();
        pattern->collectPatternAttributes(attributes);

        auto* resources = SVGResourcesCache::cachedResourcesForRenderer(*current);
        ASSERT_IMPLIES(resources && resources->linkedResource(), is<LegacyRenderSVGResourcePattern>(resources->linkedResource()));
        current = resources ? downcast<LegacyRenderSVGResourcePattern>(resources->linkedResource()) : nullptr;
    }
}

PatternData* LegacyRenderSVGResourcePattern::buildPattern(RenderElement& renderer, OptionSet<RenderSVGResourceMode> resourceMode, GraphicsContext& context)
{
    ASSERT(!m_shouldCollectPatternAttributes);

    PatternData* currentData = m_patternMap.get(renderer);
    if (currentData && currentData->pattern)
        return currentData;

    // If we couldn't determine the pattern content element root, stop here.
    if (!m_attributes.patternContentElement())
        return nullptr;

    // An empty viewBox disables rendering.
    if (m_attributes.hasViewBox() && m_attributes.viewBox().isEmpty())
        return nullptr;

    // Compute all necessary transformations to build the tile image & the pattern.
    FloatRect tileBoundaries;
    AffineTransform tileImageTransform;
    if (!buildTileImageTransform(renderer, m_attributes, protectedPatternElement(), tileBoundaries, tileImageTransform))
        return nullptr;

    auto absoluteTransform = SVGRenderingContext::calculateTransformationToOutermostCoordinateSystem(renderer);

    // Ignore 2D rotation, as it doesn't affect the size of the tile.
    FloatSize tileScale(absoluteTransform.xScale(), absoluteTransform.yScale());

    // Scale the tile size to match the scale level of the patternTransform.
    tileScale.scale(static_cast<float>(m_attributes.patternTransform().xScale()), static_cast<float>(m_attributes.patternTransform().yScale()));

    // Build tile image.
    auto tileImage = createTileImage(context, tileBoundaries.size(), tileScale, tileImageTransform, m_attributes);
    if (!tileImage)
        return nullptr;

    auto tileImageSize = tileImage->logicalSize();

    // Compute pattern space transformation.
    auto patternData = makeUnique<PatternData>();
    patternData->transform.translate(tileBoundaries.location());
    patternData->transform.scale(tileBoundaries.size() / tileImageSize);

    AffineTransform patternTransform = m_attributes.patternTransform();
    if (!patternTransform.isIdentity())
        patternData->transform = patternTransform * patternData->transform;

    // Account for text drawing resetting the context to non-scaled, see SVGInlineTextBox::paintTextWithShadows.
    if (resourceMode.contains(RenderSVGResourceMode::ApplyToText)) {
        auto textScale = computeTextPaintingScale(renderer);
        if (textScale != 1)
            patternData->transform.scale(textScale);
    }

    // Build pattern.
    patternData->pattern = Pattern::create({ tileImage.releaseNonNull() }, { true, true, patternData->transform });

    // Various calls above may trigger invalidations in some fringe cases (ImageBuffer allocation
    // failures in the SVG image cache for example). To avoid having our PatternData deleted by
    // removeAllClientsFromCacheAndMarkForInvalidation(), we only make it visible in the cache at the very end.
    return m_patternMap.set(renderer, WTFMove(patternData)).iterator->value.get();
}

auto LegacyRenderSVGResourcePattern::applyResource(RenderElement& renderer, const RenderStyle& style, GraphicsContext*& context, OptionSet<RenderSVGResourceMode> resourceMode) -> OptionSet<ApplyResult>
{
    ASSERT(context);
    ASSERT(!resourceMode.isEmpty());

    if (m_shouldCollectPatternAttributes) {
        protectedPatternElement()->synchronizeAllAttributes();

        m_attributes = PatternAttributes();
        collectPatternAttributes(m_attributes);
        m_shouldCollectPatternAttributes = false;
    }
    
    // Spec: When the geometry of the applicable element has no width or height and objectBoundingBox is specified,
    // then the given effect (e.g. a gradient or a filter) will be ignored.
    FloatRect objectBoundingBox = renderer.objectBoundingBox();
    if (m_attributes.patternUnits() == SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX && objectBoundingBox.isEmpty())
        return { };

    PatternData* patternData = buildPattern(renderer, resourceMode, *context);
    if (!patternData)
        return { };

    // Draw pattern
    context->save();

    Ref svgStyle = style.svgStyle();

    if (resourceMode.contains(RenderSVGResourceMode::ApplyToFill)) {
        context->setAlpha(svgStyle->fillOpacity());
        context->setFillPattern(*patternData->pattern);
        context->setFillRule(svgStyle->fillRule());
    } else if (resourceMode.contains(RenderSVGResourceMode::ApplyToStroke)) {
        if (svgStyle->vectorEffect() == VectorEffect::NonScalingStroke)
            patternData->pattern->setPatternSpaceTransform(transformOnNonScalingStroke(&renderer, patternData->transform));
        context->setAlpha(svgStyle->strokeOpacity());
        context->setStrokePattern(*patternData->pattern);
        SVGRenderSupport::applyStrokeStyleToContext(*context, style, renderer);
    }

    if (resourceMode.contains(RenderSVGResourceMode::ApplyToText)) {
        if (resourceMode.contains(RenderSVGResourceMode::ApplyToFill)) {
            context->setTextDrawingMode(TextDrawingMode::Fill);

#if USE(CG)
            context->applyFillPattern();
#endif
        } else if (resourceMode.contains(RenderSVGResourceMode::ApplyToStroke)) {
            context->setTextDrawingMode(TextDrawingMode::Stroke);

#if USE(CG)
            context->applyStrokePattern();
#endif
        }
    }

    return { ApplyResult::ResourceApplied };
}

void LegacyRenderSVGResourcePattern::postApplyResource(RenderElement&, GraphicsContext*& context, OptionSet<RenderSVGResourceMode> resourceMode, const Path* path, const RenderElement* shape)
{
    ASSERT(context);
    ASSERT(!resourceMode.isEmpty());
    fillAndStrokePathOrShape(*context, resourceMode, path, shape);
    context->restore();
}

static inline FloatRect calculatePatternBoundaries(const PatternAttributes& attributes,
                                                   const FloatRect& objectBoundingBox,
                                                   const SVGPatternElement& patternElement)
{
    return SVGLengthContext::resolveRectangle(&patternElement, attributes.patternUnits(), objectBoundingBox, attributes.x(), attributes.y(), attributes.width(), attributes.height());
}

bool LegacyRenderSVGResourcePattern::buildTileImageTransform(RenderElement& renderer,
                                                       const PatternAttributes& attributes,
                                                       const SVGPatternElement& patternElement,
                                                       FloatRect& patternBoundaries,
                                                       AffineTransform& tileImageTransform) const
{
    FloatRect objectBoundingBox = renderer.objectBoundingBox();
    patternBoundaries = calculatePatternBoundaries(attributes, objectBoundingBox, patternElement); 
    if (patternBoundaries.width() <= 0 || patternBoundaries.height() <= 0)
        return false;

    AffineTransform viewBoxCTM = SVGFitToViewBox::viewBoxToViewTransform(attributes.viewBox(), attributes.preserveAspectRatio(), patternBoundaries.width(), patternBoundaries.height());

    // Apply viewBox/objectBoundingBox transformations.
    if (!viewBoxCTM.isIdentity())
        tileImageTransform = viewBoxCTM;
    else if (attributes.patternContentUnits() == SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX)
        tileImageTransform.scale(objectBoundingBox.width(), objectBoundingBox.height());

    return true;
}

RefPtr<ImageBuffer> LegacyRenderSVGResourcePattern::createTileImage(GraphicsContext& context, const FloatSize& size, const FloatSize& scale, const AffineTransform& tileImageTransform, const PatternAttributes& attributes) const
{
    // This is equivalent to making createImageBuffer() use roundedIntSize().
    auto roundedUnscaledImageBufferSize = [](const FloatSize& size, const FloatSize& scale) -> FloatSize {
        auto scaledSize = size * scale;
        return size - (expandedIntSize(scaledSize) - roundedIntSize(scaledSize)) * (scaledSize - flooredIntSize(scaledSize)) / scale;
    };

    auto tileSize = roundedUnscaledImageBufferSize(size, scale);

    // FIXME: Use createImageBuffer(rect, scale), delete the above calculations and fix 'tileImageTransform'
    auto tileImage = context.createScaledImageBuffer(tileSize, scale);
    if (!tileImage)
        return nullptr;

    GraphicsContext& tileImageContext = tileImage->context();

    // Apply tile image transformations.
    if (!tileImageTransform.isIdentity())
        tileImageContext.concatCTM(tileImageTransform);

    AffineTransform contentTransformation;
    if (attributes.patternContentUnits() == SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX)
        contentTransformation = tileImageTransform;

    // Draw the content into the ImageBuffer.
    for (auto& child : childrenOfType<SVGElement>(Ref { *attributes.patternContentElement() })) {
        if (!child.renderer())
            continue;
        if (child.renderer()->needsLayout())
            return nullptr;
        SVGRenderingContext::renderSubtreeToContext(tileImageContext, *child.renderer(), contentTransformation);
    }

    return tileImage;
}

}
