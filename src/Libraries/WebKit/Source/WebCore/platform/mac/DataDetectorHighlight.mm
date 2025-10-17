/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#import "config.h"
#import "DataDetectorHighlight.h"

#if ENABLE(DATA_DETECTION) && PLATFORM(MAC)

#import "Chrome.h"
#import "ChromeClient.h"
#import "FloatRect.h"
#import "GraphicsContext.h"
#import "GraphicsLayer.h"
#import "GraphicsLayerFactory.h"
#import "ImageBuffer.h"
#import <wtf/Seconds.h>
#import <pal/mac/DataDetectorsSoftLink.h>

namespace WebCore {

constexpr Seconds highlightFadeAnimationDuration = 300_ms;
constexpr double highlightFadeAnimationFrameRate = 30;

Ref<DataDetectorHighlight> DataDetectorHighlight::createForSelection(DataDetectorHighlightClient& client, RetainPtr<DDHighlightRef>&& ddHighlight, SimpleRange&& range)
{
    return adoptRef(*new DataDetectorHighlight(client, DataDetectorHighlight::Type::Selection, WTFMove(ddHighlight), { WTFMove(range) }));
}

Ref<DataDetectorHighlight> DataDetectorHighlight::createForTelephoneNumber(DataDetectorHighlightClient& client, RetainPtr<DDHighlightRef>&& ddHighlight, SimpleRange&& range)
{
    return adoptRef(*new DataDetectorHighlight(client, DataDetectorHighlight::Type::TelephoneNumber, WTFMove(ddHighlight), { WTFMove(range) }));
}

Ref<DataDetectorHighlight> DataDetectorHighlight::createForImageOverlay(DataDetectorHighlightClient& client, RetainPtr<DDHighlightRef>&& ddHighlight, SimpleRange&& range)
{
    return adoptRef(*new DataDetectorHighlight(client, DataDetectorHighlight::Type::ImageOverlay, WTFMove(ddHighlight), { WTFMove(range) }));
}

#if ENABLE(UNIFIED_PDF_DATA_DETECTION)
Ref<DataDetectorHighlight> DataDetectorHighlight::createForPDFSelection(DataDetectorHighlightClient& client, RetainPtr<DDHighlightRef>&& ddHighlight)
{
    return adoptRef(*new DataDetectorHighlight(client, DataDetectorHighlight::Type::PDFSelection, WTFMove(ddHighlight), { }));
}
#endif

DataDetectorHighlight::DataDetectorHighlight(DataDetectorHighlightClient& client, Type type, RetainPtr<DDHighlightRef>&& ddHighlight, std::optional<SimpleRange>&& range)
    : m_client(client)
    , m_range(WTFMove(range))
    , m_graphicsLayer(client.createGraphicsLayer(*this).releaseNonNull())
    , m_type(type)
    , m_fadeAnimationTimer(*this, &DataDetectorHighlight::fadeAnimationTimerFired)
{
    ASSERT(ddHighlight);
    ASSERT(isRangeSupportingType() == m_range.has_value());

    m_graphicsLayer->setDrawsContent(true);

    setHighlight(ddHighlight.get());

    layer().setOpacity(0);
}

DataDetectorHighlight::~DataDetectorHighlight()
{
    invalidate();
}

void DataDetectorHighlight::setHighlight(DDHighlightRef highlight)
{
    if (!PAL::isDataDetectorsFrameworkAvailable())
        return;

    if (!m_client)
        return;

    m_highlight = highlight;

    if (!m_highlight)
        return;

    CGRect highlightBoundingRect = PAL::softLink_DataDetectors_DDHighlightGetBoundingRect(m_highlight.get());
    m_graphicsLayer->setPosition(FloatPoint(highlightBoundingRect.origin));
    m_graphicsLayer->setSize(FloatSize(highlightBoundingRect.size));

    m_graphicsLayer->setNeedsDisplay();
}

void DataDetectorHighlight::invalidate()
{
    m_fadeAnimationTimer.stop();
    layer().removeFromParent();
    m_client = nullptr;
}

void DataDetectorHighlight::notifyFlushRequired(const GraphicsLayer*)
{
    if (!m_client)
        return;

    m_client->scheduleRenderingUpdate(RenderingUpdateStep::LayerFlush);
}

void DataDetectorHighlight::paintContents(const GraphicsLayer*, GraphicsContext& graphicsContext, const FloatRect&, OptionSet<GraphicsLayerPaintBehavior>)
{
    if (!PAL::isDataDetectorsFrameworkAvailable())
        return;

    if (!highlight())
        return;

    CGRect highlightBoundingRect = PAL::softLink_DataDetectors_DDHighlightGetBoundingRect(highlight());
    highlightBoundingRect.origin = CGPointZero;

    auto imageBuffer = graphicsContext.createImageBuffer(FloatSize(highlightBoundingRect.size), deviceScaleFactor(), DestinationColorSpace::SRGB(), graphicsContext.renderingMode(), RenderingMethod::Local);
    if (!imageBuffer)
        return;

    CGContextRef cgContext = imageBuffer->context().platformContext();

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CGLayerRef highlightLayer = PAL::softLink_DataDetectors_DDHighlightGetLayerWithContext(highlight(), cgContext);
ALLOW_DEPRECATED_DECLARATIONS_END

    CGContextDrawLayerInRect(cgContext, highlightBoundingRect, highlightLayer);

    graphicsContext.drawConsumingImageBuffer(WTFMove(imageBuffer), highlightBoundingRect);
}

float DataDetectorHighlight::deviceScaleFactor() const
{
    if (!m_client)
        return 1;

    return m_client->deviceScaleFactor();
}

bool DataDetectorHighlight::isRangeSupportingType() const
{
#if ENABLE(UNIFIED_PDF_DATA_DETECTION)
    static constexpr OptionSet rangeSupportingHighlightTypes {
        DataDetectorHighlight::Type::TelephoneNumber,
        DataDetectorHighlight::Type::Selection,
        DataDetectorHighlight::Type::ImageOverlay,
    };

    return rangeSupportingHighlightTypes.contains(m_type);
#endif
    return true;
}

const SimpleRange& DataDetectorHighlight::range() const
{
    ASSERT(isRangeSupportingType());

    return *m_range;
}

void DataDetectorHighlight::fadeAnimationTimerFired()
{
    float animationProgress = (WallTime::now() - m_fadeAnimationStartTime) / highlightFadeAnimationDuration;
    animationProgress = std::min<float>(animationProgress, 1.0);

    float opacity = (m_fadeAnimationState == FadeAnimationState::FadingIn) ? animationProgress : 1 - animationProgress;
    layer().setOpacity(opacity);

    if (animationProgress == 1.0) {
        m_fadeAnimationTimer.stop();

        bool wasFadingOut = m_fadeAnimationState == FadeAnimationState::FadingOut;
        m_fadeAnimationState = FadeAnimationState::NotAnimating;

        if (wasFadingOut)
            didFinishFadeOutAnimation();
    }
}

void DataDetectorHighlight::dismissImmediately()
{
    layer().setOpacity(0);

    if (m_fadeAnimationTimer.isActive())
        m_fadeAnimationTimer.stop();

    m_fadeAnimationState = FadeAnimationState::NotAnimating;
    didFinishFadeOutAnimation();
}

void DataDetectorHighlight::fadeIn()
{
    if (m_fadeAnimationState == FadeAnimationState::FadingIn && m_fadeAnimationTimer.isActive())
        return;

    m_fadeAnimationState = FadeAnimationState::FadingIn;
    startFadeAnimation();
}

void DataDetectorHighlight::fadeOut()
{
    if (m_fadeAnimationState == FadeAnimationState::FadingOut && m_fadeAnimationTimer.isActive())
        return;

    m_fadeAnimationState = FadeAnimationState::FadingOut;
    startFadeAnimation();
}

void DataDetectorHighlight::startFadeAnimation()
{
    m_fadeAnimationStartTime = WallTime::now();
    m_fadeAnimationTimer.startRepeating(1_s / highlightFadeAnimationFrameRate);
}

void DataDetectorHighlight::didFinishFadeOutAnimation()
{
    if (!m_client)
        return;

    if (m_client->activeHighlight() == this)
        return;

    layer().removeFromParent();
}

bool areEquivalent(const DataDetectorHighlight* a, const DataDetectorHighlight* b)
{
    if (a == b)
        return true;

    if (!a || !b)
        return false;

    if (a->type() != b->type())
        return false;

    return !a->isRangeSupportingType() || a->range() == b->range();
}

} // namespace WebCore

#endif // ENABLE(DATA_DETECTION) && PLATFORM(MAC)
