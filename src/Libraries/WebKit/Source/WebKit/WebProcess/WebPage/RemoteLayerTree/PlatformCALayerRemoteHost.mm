/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#import "PlatformCALayerRemoteHost.h"

#import "RemoteLayerTreeContext.h"

namespace WebKit {

Ref<PlatformCALayerRemote> PlatformCALayerRemoteHost::create(WebCore::LayerHostingContextIdentifier identifier, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext& context)
{
    auto layer = adoptRef(*new PlatformCALayerRemoteHost(identifier, owner, context));
    context.layerDidEnterContext(layer.get(), layer->layerType());
    return WTFMove(layer);
}

PlatformCALayerRemoteHost::PlatformCALayerRemoteHost(WebCore::LayerHostingContextIdentifier identifier, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext& context)
    : PlatformCALayerRemote(WebCore::PlatformCALayer::LayerType::LayerTypeHost, owner, context)
    , m_identifier(identifier)
{
}

void PlatformCALayerRemoteHost::populateCreationProperties(RemoteLayerTreeTransaction::LayerCreationProperties& properties, const RemoteLayerTreeContext& context, WebCore::PlatformCALayer::LayerType type)
{
    PlatformCALayerRemote::populateCreationProperties(properties, context, type);
    ASSERT(std::holds_alternative<RemoteLayerTreeTransaction::LayerCreationProperties::NoAdditionalData>(properties.additionalData));
    properties.additionalData = m_identifier;
}

} // namespace WebKit
