/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#import "ModelPresentationManagerProxy.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(MODEL_PROCESS)

#import "UIKitSPI.h"
#import "WKPageHostedModelView.h"
#import "WebPageProxy.h"
#import <wtf/RefPtr.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ModelPresentationManagerProxy);

ModelPresentationManagerProxy::ModelPresentationManagerProxy(WebPageProxy& page)
    : m_page(page)
{
}

ModelPresentationManagerProxy::~ModelPresentationManagerProxy() = default;

RetainPtr<WKPageHostedModelView> ModelPresentationManagerProxy::setUpModelView(Ref<WebCore::ModelContext> modelContext)
{
    RefPtr webPageProxy = m_page.get();
    if (!webPageProxy)
        return nil;

    auto& modelPresentation = ensureModelPresentation(modelContext, *webPageProxy);
    return modelPresentation.pageHostedModelView;
}

void ModelPresentationManagerProxy::invalidateModel(const WebCore::PlatformLayerIdentifier& layerIdentifier)
{
    m_modelPresentations.remove(layerIdentifier);
    RELEASE_LOG_INFO(ModelElement, "%p - ModelPresentationManagerProxy removed model presentation for layer ID: %" PRIu64, this, layerIdentifier.object().toRawValue());
}

void ModelPresentationManagerProxy::invalidateAllModels()
{
    m_modelPresentations.clear();
    RELEASE_LOG_INFO(ModelElement, "%p - ModelPresentationManagerProxy removed all model presentations", this);
}

ModelPresentationManagerProxy::ModelPresentation& ModelPresentationManagerProxy::ensureModelPresentation(Ref<WebCore::ModelContext> modelContext, const WebPageProxy& webPageProxy)
{
    auto layerIdentifier = modelContext->modelLayerIdentifier();
    if (m_modelPresentations.contains(layerIdentifier)) {
        // Update the existing ModelPresentation
        ModelPresentation& modelPresentation = *(m_modelPresentations.get(layerIdentifier));
        if (modelPresentation.modelContext->modelContentsLayerHostingContextIdentifier() != modelContext->modelContentsLayerHostingContextIdentifier()) {
            modelPresentation.remoteModelView = adoptNS([[_UIRemoteView alloc] initWithFrame:CGRectZero pid:webPageProxy.legacyMainFrameProcessID() contextID:modelContext->modelContentsLayerHostingContextIdentifier().toRawValue()]);
            [modelPresentation.pageHostedModelView setRemoteModelView:modelPresentation.remoteModelView.get()];
            RELEASE_LOG_INFO(ModelElement, "%p - ModelPresentationManagerProxy updated model view for element: %" PRIu64, this, layerIdentifier.object().toRawValue());
        }
        modelPresentation.modelContext = modelContext;
    } else {
        auto pageHostedModelView = adoptNS([[WKPageHostedModelView alloc] init]);
        auto remoteModelView = adoptNS([[_UIRemoteView alloc] initWithFrame:CGRectZero pid:webPageProxy.legacyMainFrameProcessID() contextID:modelContext->modelContentsLayerHostingContextIdentifier().toRawValue()]);
        [pageHostedModelView setRemoteModelView:remoteModelView.get()];
        auto modelPresentation = ModelPresentation {
            .modelContext = modelContext,
            .remoteModelView = remoteModelView,
            .pageHostedModelView = pageHostedModelView,
        };
        m_modelPresentations.add(layerIdentifier, makeUniqueRef<ModelPresentationManagerProxy::ModelPresentation>(WTFMove(modelPresentation)));
        RELEASE_LOG_INFO(ModelElement, "%p - ModelPresentationManagerProxy created new model presentation for element: %" PRIu64, this, layerIdentifier.object().toRawValue());
    }

    return *(m_modelPresentations.get(layerIdentifier));
}

}

#endif // PLATFORM(IOS_FAMILY) && ENABLE(MODEL_PROCESS)
