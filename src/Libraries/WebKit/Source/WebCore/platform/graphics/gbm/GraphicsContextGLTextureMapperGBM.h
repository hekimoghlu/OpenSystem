/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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

#if ENABLE(WEBGL) && USE(COORDINATED_GRAPHICS) && USE(GBM)
#include "GraphicsContextGLTextureMapperANGLE.h"
#include "GraphicsLayerContentsDisplayDelegate.h"

typedef void* EGLImageKHR;
struct gbm_bo;

namespace WebCore {
class DMABufBuffer;

class GraphicsContextGLTextureMapperGBM final : public GraphicsContextGLTextureMapperANGLE {
public:
    static RefPtr<GraphicsContextGLTextureMapperGBM> create(GraphicsContextGLAttributes&&, RefPtr<GraphicsLayerContentsDisplayDelegate>&& = nullptr);
    virtual ~GraphicsContextGLTextureMapperGBM();

    void prepareForDisplayWithFinishedSignal(Function<void()>&&);
    DMABufBuffer* displayBuffer() { return m_displayBuffer.dmabuf.get(); }

private:
    GraphicsContextGLTextureMapperGBM(GraphicsContextGLAttributes&&, RefPtr<GraphicsLayerContentsDisplayDelegate>&&);

    bool platformInitialize() override;
    bool platformInitializeExtensions() override;
    bool reshapeDrawingBuffer() override;
    void prepareForDisplay() override;

    void freeDrawingBuffers();
    bool bindNextDrawingBuffer();

    struct DrawingBuffer {
        RefPtr<DMABufBuffer> dmabuf;
        EGLImageKHR image { nullptr };
    };
    DrawingBuffer createDrawingBuffer() const;

    struct {
        uint32_t fourcc { 0 };
        Vector<uint64_t, 1> modifiers;
    } m_drawingBufferFormat;

    DrawingBuffer m_drawingBuffer;
    DrawingBuffer m_displayBuffer;
};

} // namespace WebCore

#endif // ENABLE(WEBGL) && USE(COORDINATED_GRAPHICS) && USE(GBM)
