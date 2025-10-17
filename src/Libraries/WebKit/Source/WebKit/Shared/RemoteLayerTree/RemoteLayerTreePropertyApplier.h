/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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

#include "RemoteLayerTreeNode.h"
#include "RemoteLayerTreeTransaction.h"
#include <wtf/HashMap.h>

namespace WebKit {

class RemoteLayerTreeHost;

class RemoteLayerTreePropertyApplier {
public:
    using RelatedLayerMap = HashMap<WebCore::PlatformLayerIdentifier, Ref<RemoteLayerTreeNode>>;
    
    static void applyHierarchyUpdates(RemoteLayerTreeNode&, const LayerProperties&, const RelatedLayerMap&);
    static void applyProperties(RemoteLayerTreeNode&, RemoteLayerTreeHost*, const LayerProperties&, const RelatedLayerMap&, LayerContentsType);
    static void applyPropertiesToLayer(CALayer *, RemoteLayerTreeNode*, RemoteLayerTreeHost*, const LayerProperties&, LayerContentsType);

private:
    static void updateMask(RemoteLayerTreeNode&, const LayerProperties&, const RelatedLayerMap&);
#if PLATFORM(IOS_FAMILY)
    static void applyPropertiesToUIView(UIView *, const LayerProperties&, const RelatedLayerMap&);
#endif
};

} // namespace WebKit
