/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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

#if PLATFORM(IOS_FAMILY) && ENABLE(MODEL_PROCESS)

#import <WebCore/ModelContext.h>
#import <wtf/RefCounted.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/UniqueRef.h>

OBJC_CLASS WKPageHostedModelView;
OBJC_CLASS UIView;
OBJC_CLASS _UIRemoteView;

namespace WebKit {

class WebPageProxy;

class ModelPresentationManagerProxy : public RefCounted<ModelPresentationManagerProxy> {
    WTF_MAKE_TZONE_ALLOCATED(ModelPresentationManagerProxy);
public:
    static Ref<ModelPresentationManagerProxy> create(WebPageProxy& page)
    {
        return adoptRef(*new ModelPresentationManagerProxy(page));
    }

    virtual ~ModelPresentationManagerProxy();

    RetainPtr<WKPageHostedModelView> setUpModelView(Ref<WebCore::ModelContext>);
    void invalidateModel(const WebCore::PlatformLayerIdentifier&);
    void invalidateAllModels();

private:
    explicit ModelPresentationManagerProxy(WebPageProxy&);

    struct ModelPresentation {
        WTF_MAKE_FAST_ALLOCATED;

    public:
        Ref<WebCore::ModelContext> modelContext;
        RetainPtr<_UIRemoteView> remoteModelView;
        RetainPtr<WKPageHostedModelView> pageHostedModelView;
    };

    ModelPresentation& ensureModelPresentation(Ref<WebCore::ModelContext>, const WebPageProxy&);

    HashMap<WebCore::PlatformLayerIdentifier, UniqueRef<ModelPresentation>> m_modelPresentations;
    WeakPtr<WebPageProxy> m_page;
};

}

#endif // PLATFORM(IOS_FAMILY) && ENABLE(MODEL_PROCESS)
