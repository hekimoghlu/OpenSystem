/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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
#include "CairoPaintingOperation.h"
#include "RefPtrCairo.h"
#include <memory>
#include <wtf/TZoneMalloc.h>

typedef struct _cairo_surface cairo_surface_t;

namespace WebCore {
class CoordinatedTileBuffer;
class GraphicsContext;

namespace Cairo {

class PaintingContext {
    WTF_MAKE_TZONE_ALLOCATED(PaintingContext);
public:
    template<typename T>
    static void paint(WebCore::CoordinatedTileBuffer& buffer, const T& paintFunctor)
    {
        auto paintingContext = PaintingContext::createForPainting(buffer);
        paintFunctor(paintingContext->graphicsContext());
    }

    template<typename T>
    static void record(PaintingOperations& paintingOperations, const T& recordFunctor)
    {
        auto recordingContext = PaintingContext::createForRecording(paintingOperations);
        recordFunctor(recordingContext->graphicsContext());
    }

    static void replay(WebCore::CoordinatedTileBuffer& buffer, const PaintingOperations& paintingOperations)
    {
        auto paintingContext = PaintingContext::createForPainting(buffer);
        paintingContext->replay(paintingOperations);
    }

    ~PaintingContext();

    WebCore::GraphicsContext& graphicsContext() { return *m_graphicsContext; }

private:
    static std::unique_ptr<PaintingContext> createForPainting(WebCore::CoordinatedTileBuffer&);
    static std::unique_ptr<PaintingContext> createForRecording(PaintingOperations&);

    explicit PaintingContext(WebCore::CoordinatedTileBuffer&);
    explicit PaintingContext(PaintingOperations&);

    void replay(const PaintingOperations&);

    RefPtr<cairo_surface_t> m_surface;
    std::unique_ptr<WebCore::GraphicsContext> m_graphicsContext;
#if ASSERT_ENABLED
    bool m_deletionComplete { false };
#endif
};

} // namespace Cairo
} // namespace WebCore

#endif // USE(CAIRO)
