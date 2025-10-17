/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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

#include "PlatformCALayerRemote.h"

#if ENABLE(MODEL_ELEMENT)

namespace WebKit {

class PlatformCALayerRemoteModelHosting final : public PlatformCALayerRemote {
    friend class PlatformCALayerRemote;
public:
    static Ref<PlatformCALayerRemote> create(Ref<WebCore::Model>, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);

    virtual ~PlatformCALayerRemoteModelHosting();

private:
    PlatformCALayerRemoteModelHosting(Ref<WebCore::Model>, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);

    Type type() const final { return Type::RemoteModel; }

    Ref<WebCore::PlatformCALayer> clone(WebCore::PlatformCALayerClient* owner) const override;
    
    void populateCreationProperties(RemoteLayerTreeTransaction::LayerCreationProperties&, const RemoteLayerTreeContext&, WebCore::PlatformCALayer::LayerType) override;
    
    void dumpAdditionalProperties(TextStream&, OptionSet<WebCore::PlatformLayerTreeAsTextFlags>) final;

    Ref<WebCore::Model> m_model;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_PLATFORM_CALAYER(WebKit::PlatformCALayerRemoteModelHosting, type() == WebCore::PlatformCALayer::Type::RemoteModel)

#endif // ENABLE(MODEL_ELEMENT)
