/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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
#include "FloatRect.h"
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {
class GraphicsLayer;
class TransformationMatrix;

class CoordinatedAnimatedBackingStoreClient final : public ThreadSafeRefCounted<CoordinatedAnimatedBackingStoreClient> {
public:
    static Ref<CoordinatedAnimatedBackingStoreClient> create(GraphicsLayer&);
    ~CoordinatedAnimatedBackingStoreClient() = default;

    void invalidate();
    void update(const FloatRect& visibleRect, const FloatRect& coverRect, const FloatSize&, float contentsScale);
    void requestBackingStoreUpdateIfNeeded(const TransformationMatrix&);

private:
    explicit CoordinatedAnimatedBackingStoreClient(GraphicsLayer&);

    GraphicsLayer* m_layer { nullptr };
    FloatRect m_visibleRect;
    FloatRect m_coverRect;
    FloatSize m_size;
    float m_contentsScale { 1 };
};

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
