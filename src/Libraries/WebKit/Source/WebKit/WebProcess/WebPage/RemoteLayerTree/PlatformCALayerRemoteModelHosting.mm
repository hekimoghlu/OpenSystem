/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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
#import "PlatformCALayerRemoteModelHosting.h"

#if ENABLE(MODEL_ELEMENT)

#import "RemoteLayerTreeContext.h"
#import "WebProcess.h"
#import <WebCore/GraphicsLayerCA.h>
#import <WebCore/PlatformCALayerCocoa.h>
#import <wtf/RetainPtr.h>

namespace WebKit {

Ref<PlatformCALayerRemote> PlatformCALayerRemoteModelHosting::create(Ref<WebCore::Model> model, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext& context)
{
    auto layer = adoptRef(*new PlatformCALayerRemoteModelHosting(model, owner, context));
    context.layerDidEnterContext(layer.get(), layer->layerType());
    return WTFMove(layer);
}

PlatformCALayerRemoteModelHosting::PlatformCALayerRemoteModelHosting(Ref<WebCore::Model> model, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext& context)
    : PlatformCALayerRemote(WebCore::PlatformCALayer::LayerType::LayerTypeModelLayer, owner, context)
    , m_model(model)
{
}

PlatformCALayerRemoteModelHosting::~PlatformCALayerRemoteModelHosting() = default;

Ref<WebCore::PlatformCALayer> PlatformCALayerRemoteModelHosting::clone(WebCore::PlatformCALayerClient* owner) const
{
    auto clone = adoptRef(*new PlatformCALayerRemoteModelHosting(m_model, owner, *context()));
    context()->layerDidEnterContext(clone.get(), clone->layerType());

    updateClonedLayerProperties(clone.get(), false);

    clone->setClonedLayer(this);
    return WTFMove(clone);
}

void PlatformCALayerRemoteModelHosting::populateCreationProperties(RemoteLayerTreeTransaction::LayerCreationProperties& properties, const RemoteLayerTreeContext& context, PlatformCALayer::LayerType type)
{
    PlatformCALayerRemote::populateCreationProperties(properties, context, type);
    ASSERT(std::holds_alternative<RemoteLayerTreeTransaction::LayerCreationProperties::NoAdditionalData>(properties.additionalData));
    properties.additionalData = m_model;
}

void PlatformCALayerRemoteModelHosting::dumpAdditionalProperties(TextStream& ts, OptionSet<WebCore::PlatformLayerTreeAsTextFlags> flags)
{
    if (flags.contains(WebCore::PlatformLayerTreeAsTextFlags::IncludeModels))
        ts << indent << "(model data size " << m_model->data()->size() << ")\n";
}

} // namespace WebKit

#endif // ENABLE(MODEL_ELEMENT)
