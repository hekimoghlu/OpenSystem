/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#include "GLFence.h"
#include "TextureMapperFlags.h"
#include "TextureMapperPlatformLayer.h"
#include <wtf/OptionSet.h>

namespace WebCore {

class CoordinatedPlatformLayerBuffer : public TextureMapperPlatformLayer {
    WTF_MAKE_NONCOPYABLE(CoordinatedPlatformLayerBuffer);
    WTF_MAKE_FAST_ALLOCATED();
public:
    enum class Type : uint8_t {
        RGB,
        YUV,
        ExternalOES,
        HolePunch,
        Video,
        DMABuf,
        NativeImage
    };

    virtual ~CoordinatedPlatformLayerBuffer() = default;

    Type type() const { return m_type; }
    const IntSize& size() const { return m_size; }
    OptionSet<TextureMapperFlags> flags() const { return m_flags; }

    void waitForContentsIfNeeded()
    {
        if (auto fence = WTFMove(m_fence))
            fence->serverWait();
    }

protected:
    CoordinatedPlatformLayerBuffer(Type type, const IntSize& size, OptionSet<TextureMapperFlags> flags, std::unique_ptr<GLFence>&& fence)
        : m_type(type)
        , m_size(size)
        , m_flags(flags)
        , m_fence(WTFMove(fence))
    {
    }

    bool isHolePunchBuffer() const final { return m_type == Type::HolePunch; }

    Type m_type;
    IntSize m_size;
    OptionSet<TextureMapperFlags> m_flags;
    std::unique_ptr<GLFence> m_fence;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_COORDINATED_PLATFORM_LAYER_BUFFER_TYPE(ToValueTypeName, TypeEnumValue) \
    SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName)              \
    static bool isType(const WebCore::CoordinatedPlatformLayerBuffer& buffer) { return buffer.type() == WebCore::CoordinatedPlatformLayerBuffer::TypeEnumValue; } \
    SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(COORDINATED_GRAPHICS)
