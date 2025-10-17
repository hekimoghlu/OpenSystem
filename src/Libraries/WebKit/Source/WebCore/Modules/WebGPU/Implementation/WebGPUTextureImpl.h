/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUIntegralTypes.h"
#include "WebGPUPtr.h"
#include "WebGPUTexture.h"
#include "WebGPUTextureDimension.h"
#include "WebGPUTextureFormat.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class TextureImpl final : public Texture {
    WTF_MAKE_TZONE_ALLOCATED(TextureImpl);
public:
    static Ref<TextureImpl> create(WebGPUPtr<WGPUTexture>&& texture, TextureFormat format, TextureDimension dimension, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new TextureImpl(WTFMove(texture), format, dimension, convertToBackingContext));
    }

    virtual ~TextureImpl();

private:
    friend class DowncastConvertToBackingContext;

    TextureImpl(WebGPUPtr<WGPUTexture>&&, TextureFormat, TextureDimension, ConvertToBackingContext&);

    TextureImpl(const TextureImpl&) = delete;
    TextureImpl(TextureImpl&&) = delete;
    TextureImpl& operator=(const TextureImpl&) = delete;
    TextureImpl& operator=(TextureImpl&&) = delete;

    WGPUTexture backing() const { return m_backing.get(); }

    RefPtr<TextureView> createView(const std::optional<TextureViewDescriptor>&) final;

    void destroy() final;
    void undestroy() final;

    void setLabelInternal(const String&) final;

    TextureFormat m_format { TextureFormat::Rgba8unorm };
    TextureDimension m_dimension { TextureDimension::_2d };

    WebGPUPtr<WGPUTexture> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
