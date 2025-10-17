/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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

#if USE(COORDINATED_GRAPHICS)
#include "GraphicsLayerContentsDisplayDelegate.h"

namespace WebCore {

class GraphicsLayer;

class GraphicsLayerAsyncContentsDisplayDelegateCoordinated final : public GraphicsLayerAsyncContentsDisplayDelegate {
public:
    static Ref<GraphicsLayerAsyncContentsDisplayDelegateCoordinated> create(GraphicsLayer& layer)
    {
        return adoptRef(*new GraphicsLayerAsyncContentsDisplayDelegateCoordinated(layer));
    }
    virtual ~GraphicsLayerAsyncContentsDisplayDelegateCoordinated();

    void updateGraphicsLayer(GraphicsLayer&);

private:
    explicit GraphicsLayerAsyncContentsDisplayDelegateCoordinated(GraphicsLayer&);

    void setDisplayBuffer(std::unique_ptr<CoordinatedPlatformLayerBuffer>&&) override { RELEASE_ASSERT_NOT_REACHED(); }
    bool display(CoordinatedPlatformLayer&) override { RELEASE_ASSERT_NOT_REACHED(); }

    bool tryCopyToLayer(ImageBuffer&) override;

    Ref<GraphicsLayerContentsDisplayDelegate> m_delegate;
};

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
