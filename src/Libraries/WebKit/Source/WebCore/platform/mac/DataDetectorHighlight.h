/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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

#if ENABLE(DATA_DETECTION) && PLATFORM(MAC)

#import "GraphicsLayer.h"
#import "GraphicsLayerClient.h"
#import "SimpleRange.h"
#import "Timer.h"
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakPtr.h>

using DDHighlightRef = struct __DDHighlight*;

namespace WebCore {
class DataDetectorHighlightClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::DataDetectorHighlightClient> : std::true_type { };
}

namespace WebCore {

class DataDetectorHighlight;
class FloatRect;
class GraphicsContext;
class GraphicsLayer;

enum class RenderingUpdateStep : uint32_t;

class DataDetectorHighlightClient : public CanMakeWeakPtr<DataDetectorHighlightClient> {
public:
    WEBCORE_EXPORT virtual ~DataDetectorHighlightClient() = default;
    WEBCORE_EXPORT virtual DataDetectorHighlight* activeHighlight() const = 0;
    WEBCORE_EXPORT virtual void scheduleRenderingUpdate(OptionSet<RenderingUpdateStep>) = 0;
    WEBCORE_EXPORT virtual float deviceScaleFactor() const = 0;
    WEBCORE_EXPORT virtual RefPtr<GraphicsLayer> createGraphicsLayer(GraphicsLayerClient&) = 0;
};

class DataDetectorHighlight : public RefCountedAndCanMakeWeakPtr<DataDetectorHighlight>, private GraphicsLayerClient {
    WTF_MAKE_NONCOPYABLE(DataDetectorHighlight);
public:
    static Ref<DataDetectorHighlight> createForSelection(DataDetectorHighlightClient&, RetainPtr<DDHighlightRef>&&, SimpleRange&&);
    static Ref<DataDetectorHighlight> createForTelephoneNumber(DataDetectorHighlightClient&, RetainPtr<DDHighlightRef>&&, SimpleRange&&);
    static Ref<DataDetectorHighlight> createForImageOverlay(DataDetectorHighlightClient&, RetainPtr<DDHighlightRef>&&, SimpleRange&&);
#if ENABLE(UNIFIED_PDF_DATA_DETECTION)
    WEBCORE_EXPORT static Ref<DataDetectorHighlight> createForPDFSelection(DataDetectorHighlightClient&, RetainPtr<DDHighlightRef>&&);
#endif

    ~DataDetectorHighlight();

    void invalidate();

    DDHighlightRef highlight() const { return m_highlight.get(); }
    const SimpleRange& range() const;
    GraphicsLayer& layer() const { return m_graphicsLayer.get(); }
    Ref<GraphicsLayer> protectedLayer() const { return layer(); }

    enum class Type : uint8_t {
        None = 0,
        TelephoneNumber = 1 << 0,
        Selection = 1 << 1,
        ImageOverlay = 1 << 2,
#if ENABLE(UNIFIED_PDF_DATA_DETECTION)
        PDFSelection = 1 << 3,
#endif
    };

    Type type() const { return m_type; }
    bool isRangeSupportingType() const;

    WEBCORE_EXPORT void fadeIn();
    WEBCORE_EXPORT void fadeOut();
    WEBCORE_EXPORT void dismissImmediately();

    WEBCORE_EXPORT void setHighlight(DDHighlightRef);

private:
    DataDetectorHighlight(DataDetectorHighlightClient&, Type, RetainPtr<DDHighlightRef>&&, std::optional<SimpleRange>&&);

    // GraphicsLayerClient
    void notifyFlushRequired(const GraphicsLayer*) override;
    void paintContents(const GraphicsLayer*, GraphicsContext&, const FloatRect& inClip, OptionSet<GraphicsLayerPaintBehavior>) override;
    float deviceScaleFactor() const override;

    void fadeAnimationTimerFired();
    void startFadeAnimation();
    void didFinishFadeOutAnimation();

    WeakPtr<DataDetectorHighlightClient> m_client;
    RetainPtr<DDHighlightRef> m_highlight;
    std::optional<SimpleRange> m_range;
    Ref<GraphicsLayer> m_graphicsLayer;
    Type m_type { Type::None };

    Timer m_fadeAnimationTimer;
    WallTime m_fadeAnimationStartTime;

    enum class FadeAnimationState : uint8_t { NotAnimating, FadingIn, FadingOut };
    FadeAnimationState m_fadeAnimationState { FadeAnimationState::NotAnimating };
};

bool areEquivalent(const DataDetectorHighlight*, const DataDetectorHighlight*);

} // namespace WebCore

#endif // ENABLE(DATA_DETECTION) && PLATFORM(MAC)
