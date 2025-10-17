/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#import "RemoteLayerTreeHost.h"

#if PLATFORM(IOS_FAMILY)

#import "RemoteLayerTreeDrawingAreaProxy.h"
#import "RemoteLayerTreeViews.h"
#import "UIKitSPI.h"
#import "VideoPresentationManagerProxy.h"
#import "WKVideoView.h"
#import "WebPageProxy.h"
#import "WebPreferences.h"
#import <UIKit/UIScrollView.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)
#import "WKModelView.h"
#endif

#if ENABLE(MODEL_PROCESS)
#import "ModelPresentationManagerProxy.h"
#import "WKPageHostedModelView.h"
#endif

#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
#import "WKSeparatedImageView.h"
#endif

namespace WebKit {
using namespace WebCore;

RefPtr<RemoteLayerTreeNode> RemoteLayerTreeHost::makeNode(const RemoteLayerTreeTransaction::LayerCreationProperties& properties)
{
    auto makeWithView = [&] (RetainPtr<UIView>&& view) {
        return RemoteLayerTreeNode::create(*properties.layerID, properties.hostIdentifier(), WTFMove(view));
    };

    switch (properties.type) {
    case PlatformCALayer::LayerType::LayerTypeLayer:
    case PlatformCALayer::LayerType::LayerTypeWebLayer:
    case PlatformCALayer::LayerType::LayerTypeRootLayer:
    case PlatformCALayer::LayerType::LayerTypeSimpleLayer:
    case PlatformCALayer::LayerType::LayerTypeTiledBackingLayer:
    case PlatformCALayer::LayerType::LayerTypePageTiledBackingLayer:
    case PlatformCALayer::LayerType::LayerTypeContentsProvidedLayer:
    case PlatformCALayer::LayerType::LayerTypeHost:
        return makeWithView(adoptNS([[WKCompositingView alloc] init]));

    case PlatformCALayer::LayerType::LayerTypeTiledBackingTileLayer:
        return RemoteLayerTreeNode::createWithPlainLayer(*properties.layerID);

    case PlatformCALayer::LayerType::LayerTypeBackdropLayer:
        return makeWithView(adoptNS([[WKBackdropView alloc] init]));

#if HAVE(CORE_MATERIAL)
    case PlatformCALayer::LayerType::LayerTypeMaterialLayer:
        return makeWithView(adoptNS([[WKMaterialView alloc] init]));
#endif

    case PlatformCALayer::LayerType::LayerTypeTransformLayer:
        return makeWithView(adoptNS([[WKTransformView alloc] init]));

    case PlatformCALayer::LayerType::LayerTypeCustom:
    case PlatformCALayer::LayerType::LayerTypeAVPlayerLayer: {
        if (m_isDebugLayerTreeHost)
            return makeWithView(adoptNS([[WKCompositingView alloc] init]));

#if HAVE(AVKIT)
        if (properties.videoElementData) {
            if (auto videoManager = m_drawingArea->page() ? m_drawingArea->page()->videoPresentationManager() : nullptr) {
                m_videoLayers.add(*properties.layerID, properties.videoElementData->playerIdentifier);
                return makeWithView(videoManager->createViewWithID(properties.videoElementData->playerIdentifier, properties.hostingContextID(), properties.videoElementData->initialSize, properties.videoElementData->naturalSize, properties.hostingDeviceScaleFactor()));
            }
        }
#endif

        if (!m_drawingArea->page())
            return nullptr;

#if ENABLE(MODEL_PROCESS)
        if (auto modelContext = properties.modelContext()) {
            if (auto modelPresentationManager = m_drawingArea->page() ? m_drawingArea->page()->modelPresentationManagerProxy() : nullptr) {
                if (auto view = modelPresentationManager->setUpModelView(*modelContext)) {
                    m_modelLayers.add(modelContext->modelLayerIdentifier());
                    return makeWithView(WTFMove(view));
                }
            }
        }
#endif

        auto view = adoptNS([[WKUIRemoteView alloc] initWithFrame:CGRectZero pid:m_drawingArea->page()->legacyMainFrameProcessID() contextID:properties.hostingContextID()]);
        return makeWithView(WTFMove(view));
    }
    case PlatformCALayer::LayerType::LayerTypeShapeLayer:
        return makeWithView(adoptNS([[WKShapeView alloc] init]));

    case PlatformCALayer::LayerType::LayerTypeScrollContainerLayer:
        if (!m_isDebugLayerTreeHost)
            return makeWithView(adoptNS([[WKChildScrollView alloc] init]));
        // The debug indicator parents views under layers, which can cause crashes with UIScrollView.
        return makeWithView(adoptNS([[UIView alloc] init]));

#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
    case PlatformCALayer::LayerType::LayerTypeSeparatedImageLayer:
        return makeWithView(adoptNS([[WKSeparatedImageView alloc] init]));
#endif

#if ENABLE(MODEL_ELEMENT)
    case PlatformCALayer::LayerType::LayerTypeModelLayer:
#if ENABLE(MODEL_PROCESS)
        bool modelHandledOutOfProcess = m_drawingArea->page() && m_drawingArea->page()->preferences().modelProcessEnabled();
#else
        bool modelHandledOutOfProcess = false;
#endif

        if (!modelHandledOutOfProcess && m_drawingArea->page() && m_drawingArea->page()->preferences().modelElementEnabled()) {
            if (auto* model = std::get_if<Ref<Model>>(&properties.additionalData)) {
#if ENABLE(SEPARATED_MODEL)
                return makeWithView(adoptNS([[WKSeparatedModelView alloc] initWithModel:*model]));
#elif ENABLE(ARKIT_INLINE_PREVIEW_IOS)
                return makeWithView(adoptNS([[WKModelView alloc] initWithModel:*model layerID:*properties.layerID page:*m_drawingArea->page()]));
#endif
            }
        }
        return makeWithView(adoptNS([[WKCompositingView alloc] init]));
#endif // ENABLE(MODEL_ELEMENT)
    }
    ASSERT_NOT_REACHED();
    return nullptr;
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
