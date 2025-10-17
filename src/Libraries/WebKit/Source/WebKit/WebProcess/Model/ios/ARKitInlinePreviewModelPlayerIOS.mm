/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#import "ARKitInlinePreviewModelPlayerIOS.h"

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)

#import "WebPage.h"
#import "WebPageProxyMessages.h"
#import <WebCore/GraphicsLayer.h>
#import <wtf/WeakHashSet.h>

namespace WebKit {

Ref<ARKitInlinePreviewModelPlayerIOS> ARKitInlinePreviewModelPlayerIOS::create(WebPage& page, WebCore::ModelPlayerClient& client)
{
    return adoptRef(*new ARKitInlinePreviewModelPlayerIOS(page, client));
}

static WeakHashSet<ARKitInlinePreviewModelPlayerIOS>& instances()
{
    static NeverDestroyed<WeakHashSet<ARKitInlinePreviewModelPlayerIOS>> instances;
    return instances;
}

ARKitInlinePreviewModelPlayerIOS::ARKitInlinePreviewModelPlayerIOS(WebPage& page, WebCore::ModelPlayerClient& client)
    : ARKitInlinePreviewModelPlayer(page, client)
{
    instances().add(*this);
}

ARKitInlinePreviewModelPlayerIOS::~ARKitInlinePreviewModelPlayerIOS()
{
    instances().remove(*this);
}

ARKitInlinePreviewModelPlayerIOS* ARKitInlinePreviewModelPlayerIOS::modelPlayerForPageAndLayerID(WebPage& page, PlatformLayerIdentifier layerID)
{
    for (auto& modelPlayer : instances()) {
        if (!modelPlayer.client())
            continue;

        if (&page != modelPlayer.page())
            continue;

        if (modelPlayer.client()->modelContentsLayerID() != layerID)
            continue;

        return &modelPlayer;
    }

    return nullptr;
}

void ARKitInlinePreviewModelPlayerIOS::pageLoadedModelInlinePreview(WebPage& page, PlatformLayerIdentifier layerID)
{
    if (auto* modelPlayer = modelPlayerForPageAndLayerID(page, layerID))
        modelPlayer->client()->didFinishLoading(*modelPlayer);
}

void ARKitInlinePreviewModelPlayerIOS::pageFailedToLoadModelInlinePreview(WebPage& page, PlatformLayerIdentifier layerID, const WebCore::ResourceError& error)
{
    if (auto* modelPlayer = modelPlayerForPageAndLayerID(page, layerID))
        modelPlayer->client()->didFailLoading(*modelPlayer, error);
}

std::optional<ModelIdentifier> ARKitInlinePreviewModelPlayerIOS::modelIdentifier()
{
    if (!client())
        return { };

    if (auto layerId = client()->modelContentsLayerID())
        return { { *layerId } };

    return { };
}

// MARK: - WebCore::ModelPlayer overrides.

void ARKitInlinePreviewModelPlayerIOS::enterFullscreen()
{
    RefPtr strongPage = page();
    if (!strongPage)
        return;

    if (auto modelIdentifier = this->modelIdentifier())
        strongPage->send(Messages::WebPageProxy::TakeModelElementFullscreen(*modelIdentifier));
}

void ARKitInlinePreviewModelPlayerIOS::setInteractionEnabled(bool isInteractionEnabled)
{
    RefPtr strongPage = page();
    if (!strongPage)
        return;

    if (auto modelIdentifier = this->modelIdentifier())
        strongPage->send(Messages::WebPageProxy::ModelElementSetInteractionEnabled(*modelIdentifier, isInteractionEnabled));
}

void ARKitInlinePreviewModelPlayerIOS::handleMouseDown(const WebCore::LayoutPoint&, MonotonicTime)
{
    ASSERT_NOT_REACHED();
}

void ARKitInlinePreviewModelPlayerIOS::handleMouseMove(const WebCore::LayoutPoint&, MonotonicTime)
{
    ASSERT_NOT_REACHED();
}

void ARKitInlinePreviewModelPlayerIOS::handleMouseUp(const WebCore::LayoutPoint&, MonotonicTime)
{
    ASSERT_NOT_REACHED();
}

}
#endif
