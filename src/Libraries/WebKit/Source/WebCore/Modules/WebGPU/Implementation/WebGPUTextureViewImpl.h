/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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

#include "WebGPUPtr.h"
#include "WebGPUTextureView.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class TextureViewImpl final : public TextureView {
    WTF_MAKE_TZONE_ALLOCATED(TextureViewImpl);
public:
    static Ref<TextureViewImpl> create(WebGPUPtr<WGPUTextureView>&& textureView, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new TextureViewImpl(WTFMove(textureView), convertToBackingContext));
    }

    virtual ~TextureViewImpl();

private:
    friend class DowncastConvertToBackingContext;

    TextureViewImpl(WebGPUPtr<WGPUTextureView>&&, ConvertToBackingContext&);

    TextureViewImpl(const TextureViewImpl&) = delete;
    TextureViewImpl(TextureViewImpl&&) = delete;
    TextureViewImpl& operator=(const TextureViewImpl&) = delete;
    TextureViewImpl& operator=(TextureViewImpl&&) = delete;

    WGPUTextureView backing() const { return m_backing.get(); }

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPUTextureView> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
