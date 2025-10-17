/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#if PLATFORM(IOS_FAMILY)

#import "WebFixedPositionContent.h"
#import "WebFixedPositionContentInternal.h"

#import "WebViewInternal.h"
#import <WebCore/ChromeClient.h>
#import <WebCore/IntSize.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/ScrollingConstraints.h>
#import <WebCore/WebCoreThreadRun.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>

#import <wtf/HashMap.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>
#import <wtf/StdLibExtras.h>
#import <wtf/Threading.h>

#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <algorithm>

using namespace WebCore;

static Lock webFixedPositionContentDataLock;

struct ViewportConstrainedLayerData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    ViewportConstrainedLayerData()
        : m_enclosingAcceleratedScrollLayer(nil)
    { }
    CALayer* m_enclosingAcceleratedScrollLayer; // May be nil.
    std::unique_ptr<ViewportConstraints> m_viewportConstraints;
};

typedef HashMap<RetainPtr<CALayer>, std::unique_ptr<ViewportConstrainedLayerData>> LayerInfoMap;

struct WebFixedPositionContentData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
public:
    WebFixedPositionContentData(WebView *);
    ~WebFixedPositionContentData();
    
    WebView *m_webView;
    LayerInfoMap m_viewportConstrainedLayers;
};


WebFixedPositionContentData::WebFixedPositionContentData(WebView *webView)
    : m_webView(webView)
{
}

WebFixedPositionContentData::~WebFixedPositionContentData()
{
}

@implementation WebFixedPositionContent {
    struct WebFixedPositionContentData* _private;
}

- (id)initWithWebView:(WebView *)webView
{
    if ((self = [super init])) {
        _private = new WebFixedPositionContentData(webView);
    }
    return self;
}

- (void)dealloc
{
    delete _private;
    [super dealloc];
}

- (void)scrollOrZoomChanged:(CGRect)positionedObjectsRect
{
    Locker locker { webFixedPositionContentDataLock };

    LayerInfoMap::const_iterator end = _private->m_viewportConstrainedLayers.end();
    for (LayerInfoMap::const_iterator it = _private->m_viewportConstrainedLayers.begin(); it != end; ++it) {
        CALayer *layer = it->key.get();
        ViewportConstrainedLayerData* constraintData = it->value.get();
        const ViewportConstraints& constraints = *(constraintData->m_viewportConstraints.get());

        switch (constraints.constraintType()) {
        case ViewportConstraints::FixedPositionConstraint: {
                const FixedPositionViewportConstraints& fixedConstraints = static_cast<const FixedPositionViewportConstraints&>(constraints);

                FloatPoint layerPosition = fixedConstraints.layerPositionForViewportRect(positionedObjectsRect);
            
                CGRect layerBounds = [layer bounds];
                CGPoint anchorPoint = [layer anchorPoint];
                CGPoint newPosition = CGPointMake(layerPosition.x() - constraints.alignmentOffset().width() + anchorPoint.x * layerBounds.size.width,
                                                  layerPosition.y() - constraints.alignmentOffset().height() + anchorPoint.y * layerBounds.size.height);
                [layer setPosition:newPosition];
                break;
            }
        case ViewportConstraints::StickyPositionConstraint: {
                const StickyPositionViewportConstraints& stickyConstraints = static_cast<const StickyPositionViewportConstraints&>(constraints);

                FloatPoint layerPosition = stickyConstraints.layerPositionForConstrainingRect(positionedObjectsRect);

                CGRect layerBounds = [layer bounds];
                CGPoint anchorPoint = [layer anchorPoint];
                CGPoint newPosition = CGPointMake(layerPosition.x() - constraints.alignmentOffset().width() + anchorPoint.x * layerBounds.size.width,
                                                  layerPosition.y() - constraints.alignmentOffset().height() + anchorPoint.y * layerBounds.size.height);
                [layer setPosition:newPosition];
                break;
            }
        }
    }
}

- (void)overflowScrollPositionForLayer:(CALayer *)scrollLayer changedTo:(CGPoint)scrollPosition
{
    Locker locker { webFixedPositionContentDataLock };

    LayerInfoMap::const_iterator end = _private->m_viewportConstrainedLayers.end();
    for (LayerInfoMap::const_iterator it = _private->m_viewportConstrainedLayers.begin(); it != end; ++it) {
        CALayer *layer = it->key.get();
        ViewportConstrainedLayerData* constraintData = it->value.get();
        
        if (constraintData->m_enclosingAcceleratedScrollLayer == scrollLayer) {
            const StickyPositionViewportConstraints& stickyConstraints = static_cast<const StickyPositionViewportConstraints&>(*(constraintData->m_viewportConstraints.get()));
            FloatRect constrainingRectAtLastLayout = stickyConstraints.constrainingRectAtLastLayout();
            FloatRect scrolledConstrainingRect = FloatRect(scrollPosition.x, scrollPosition.y, constrainingRectAtLastLayout.width(), constrainingRectAtLastLayout.height());
            FloatPoint layerPosition = stickyConstraints.layerPositionForConstrainingRect(scrolledConstrainingRect);

            CGRect layerBounds = [layer bounds];
            CGPoint anchorPoint = [layer anchorPoint];
            CGPoint newPosition = CGPointMake(layerPosition.x() - stickyConstraints.alignmentOffset().width() + anchorPoint.x * layerBounds.size.width,
                                              layerPosition.y() - stickyConstraints.alignmentOffset().height() + anchorPoint.y * layerBounds.size.height);
            [layer setPosition:newPosition];
        }
    }
}

// FIXME: share code with 'sendScrollEvent'?
- (void)didFinishScrollingOrZooming
{
    WebThreadRun(^{
        if (auto* frame = [_private->m_webView _mainCoreFrame])
            frame->viewportOffsetChanged(LocalFrame::CompletedScrollOffset);
    });
}

- (void)setViewportConstrainedLayers:(WTF::HashMap<CALayer *, std::unique_ptr<WebCore::ViewportConstraints>>&)layerMap stickyContainerMap:(const WTF::HashMap<CALayer*, CALayer*>&)stickyContainers
{
    Locker locker { webFixedPositionContentDataLock };

    _private->m_viewportConstrainedLayers.clear();

    for (auto& layerAndConstraints : layerMap) {
        CALayer* layer = layerAndConstraints.key;
        auto layerData = makeUnique<ViewportConstrainedLayerData>();

        layerData->m_enclosingAcceleratedScrollLayer = stickyContainers.get(layer);
        layerData->m_viewportConstraints = WTFMove(layerAndConstraints.value);

        _private->m_viewportConstrainedLayers.set(layer, WTFMove(layerData));
    }
}

- (BOOL)hasFixedOrStickyPositionLayers
{
    Locker locker { webFixedPositionContentDataLock };
    return !_private->m_viewportConstrainedLayers.isEmpty();
}

@end

#endif // PLATFORM(IOS_FAMILY)
