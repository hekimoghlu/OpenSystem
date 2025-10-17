/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#include "config.h"
#include "RemoteGraphicsContextGLProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(WEBGL) && USE(COORDINATED_GRAPHICS) && USE(GBM)
#include <WebCore/CoordinatedPlatformLayerBufferDMABuf.h>
#include <WebCore/DMABufBuffer.h>
#include <WebCore/GraphicsLayerContentsDisplayDelegateCoordinated.h>
#include <WebCore/TextureMapperFlags.h>

namespace WebKit {
using namespace WebCore;

class RemoteGraphicsContextGLProxyGBM final : public RemoteGraphicsContextGLProxy {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteGraphicsContextGLProxyGBM);
public:
    virtual ~RemoteGraphicsContextGLProxyGBM() = default;

private:
    friend class RemoteGraphicsContextGLProxy;
    explicit RemoteGraphicsContextGLProxyGBM(const GraphicsContextGLAttributes& attributes)
        : RemoteGraphicsContextGLProxy(attributes)
        , m_layerContentsDisplayDelegate(GraphicsLayerContentsDisplayDelegateCoordinated::create())
    {
    }

    // WebCore::GraphicsContextGL
    RefPtr<GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate() final { return m_layerContentsDisplayDelegate.copyRef(); }
    void prepareForDisplay() final;

    Ref<GraphicsLayerContentsDisplayDelegate> m_layerContentsDisplayDelegate;
    RefPtr<DMABufBuffer> m_drawingBuffer;
    RefPtr<DMABufBuffer> m_displayBuffer;
};

void RemoteGraphicsContextGLProxyGBM::prepareForDisplay()
{
    if (isContextLost())
        return;

    auto sendResult = sendSync(Messages::RemoteGraphicsContextGL::PrepareForDisplay());
    if (!sendResult.succeeded()) {
        markContextLost();
        return;
    }

    auto [bufferID, bufferAttributes, fenceFD] = sendResult.takeReply();

    if (bufferAttributes || (m_drawingBuffer && m_drawingBuffer->id() == bufferID))
        std::swap(m_drawingBuffer, m_displayBuffer);

    if (bufferAttributes)
        m_displayBuffer = DMABufBuffer::create(bufferID, WTFMove(*bufferAttributes));

    if (!m_displayBuffer)
        return;

    OptionSet<TextureMapperFlags> flags = TextureMapperFlags::ShouldFlipTexture;
    if (contextAttributes().alpha)
        flags.add(TextureMapperFlags::ShouldBlend);
    m_layerContentsDisplayDelegate->setDisplayBuffer(CoordinatedPlatformLayerBufferDMABuf::create(Ref { *m_displayBuffer }, flags, WTFMove(fenceFD)));
}

Ref<RemoteGraphicsContextGLProxy> RemoteGraphicsContextGLProxy::platformCreate(const GraphicsContextGLAttributes& attributes)
{
    return adoptRef(*new RemoteGraphicsContextGLProxyGBM(attributes));
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEBGL) && USE(COORDINATED_GRAPHICS) && USE(GBM)
