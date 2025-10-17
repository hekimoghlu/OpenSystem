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

#if USE(COORDINATED_GRAPHICS) && USE(GBM)
#include "CoordinatedPlatformLayerBuffer.h"
#include <wtf/unix/UnixFileDescriptor.h>

namespace WebCore {

class DMABufBuffer;
struct DMABufBufferAttributes;

class CoordinatedPlatformLayerBufferDMABuf final : public CoordinatedPlatformLayerBuffer {
public:
    static std::unique_ptr<CoordinatedPlatformLayerBufferDMABuf> create(Ref<DMABufBuffer>&&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    static std::unique_ptr<CoordinatedPlatformLayerBufferDMABuf> create(Ref<DMABufBuffer>&&, OptionSet<TextureMapperFlags>, WTF::UnixFileDescriptor&&);
    CoordinatedPlatformLayerBufferDMABuf(Ref<DMABufBuffer>&&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    CoordinatedPlatformLayerBufferDMABuf(Ref<DMABufBuffer>&&, OptionSet<TextureMapperFlags>, WTF::UnixFileDescriptor&&);
    virtual ~CoordinatedPlatformLayerBufferDMABuf();

private:
    void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix& modelViewMatrix = TransformationMatrix(), float opacity = 1.0) override;

    std::unique_ptr<CoordinatedPlatformLayerBuffer> importDMABuf(TextureMapper&) const;
    std::unique_ptr<CoordinatedPlatformLayerBuffer> importYUV(TextureMapper&) const;

    Ref<DMABufBuffer> m_dmabuf;
    WTF::UnixFileDescriptor m_fenceFD;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_COORDINATED_PLATFORM_LAYER_BUFFER_TYPE(CoordinatedPlatformLayerBufferDMABuf, Type::DMABuf)

#endif // USE(COORDINATED_GRAPHICS) && USE(GBM)
