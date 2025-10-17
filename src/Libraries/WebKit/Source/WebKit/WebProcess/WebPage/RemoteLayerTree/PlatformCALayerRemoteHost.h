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
#pragma once

#include "PlatformCALayerRemote.h"

namespace WebKit {

class PlatformCALayerRemoteHost final : public PlatformCALayerRemote {
    friend class PlatformCALayerRemote;
public:
    static Ref<PlatformCALayerRemote> create(WebCore::LayerHostingContextIdentifier, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);
    std::optional<WebCore::LayerHostingContextIdentifier> hostingContextIdentifier() const final { return m_identifier; }

private:
    PlatformCALayerRemoteHost(WebCore::LayerHostingContextIdentifier, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);

    Type type() const final { return Type::RemoteHost; }
    void populateCreationProperties(RemoteLayerTreeTransaction::LayerCreationProperties&, const RemoteLayerTreeContext&, WebCore::PlatformCALayer::LayerType) final;

    WebCore::LayerHostingContextIdentifier m_identifier;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_PLATFORM_CALAYER(WebKit::PlatformCALayerRemoteHost, type() == WebCore::PlatformCALayer::Type::RemoteHost)
