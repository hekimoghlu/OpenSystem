/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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

#if USE(CAIRO)
#include <memory>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class CoordinatedTileBuffer;
class GraphicsLayer;
class IntRect;

namespace Cairo {

class PaintingEngine {
    WTF_MAKE_TZONE_ALLOCATED(PaintingEngine);
public:
    WEBCORE_EXPORT static std::unique_ptr<PaintingEngine> create();

    virtual ~PaintingEngine() = default;

    virtual void paint(WebCore::GraphicsLayer&, WebCore::CoordinatedTileBuffer&, const WebCore::IntRect&, const WebCore::IntRect&, const WebCore::IntRect&, float) = 0;
};

} // namespace Cairo
} // namespace WebCore

#endif // USE(CAIRO)
