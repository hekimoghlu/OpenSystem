/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#include "WebGPUTextureImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include "WebGPUTextureView.h"
#include "WebGPUTextureViewDescriptor.h"
#include "WebGPUTextureViewImpl.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextureImpl);

TextureImpl::TextureImpl(WebGPUPtr<WGPUTexture>&& texture, TextureFormat format, TextureDimension dimension, ConvertToBackingContext& convertToBackingContext)
    : m_format(format)
    , m_dimension(dimension)
    , m_backing(WTFMove(texture))
    , m_convertToBackingContext(convertToBackingContext)
{
}

TextureImpl::~TextureImpl() = default;

RefPtr<TextureView> TextureImpl::createView(const std::optional<TextureViewDescriptor>& descriptor)
{
    CString label = descriptor ? descriptor->label.utf8() : CString("");

    Ref convertToBackingContext = m_convertToBackingContext;

    WGPUTextureViewDescriptor backingDescriptor {
        .nextInChain = nullptr,
        .label = label.data(),
        .format = descriptor && descriptor->format ? convertToBackingContext->convertToBacking(*descriptor->format) : WGPUTextureFormat_Undefined,
        .dimension = descriptor && descriptor->dimension ? convertToBackingContext->convertToBacking(*descriptor->dimension) : WGPUTextureViewDimension_Undefined,
        .baseMipLevel = descriptor ? descriptor->baseMipLevel : 0,
        .mipLevelCount = descriptor && descriptor->mipLevelCount ? *descriptor->mipLevelCount : static_cast<uint32_t>(WGPU_MIP_LEVEL_COUNT_UNDEFINED),
        .baseArrayLayer = descriptor ? descriptor->baseArrayLayer : 0,
        .arrayLayerCount = descriptor && descriptor->arrayLayerCount ? *descriptor->arrayLayerCount : static_cast<uint32_t>(WGPU_ARRAY_LAYER_COUNT_UNDEFINED),
        .aspect = descriptor ? convertToBackingContext->convertToBacking(descriptor->aspect) : WGPUTextureAspect_All,
    };

    return TextureViewImpl::create(adoptWebGPU(wgpuTextureCreateView(m_backing.get(), &backingDescriptor)), convertToBackingContext);
}

void TextureImpl::destroy()
{
    wgpuTextureDestroy(m_backing.get());
}

void TextureImpl::undestroy()
{
    wgpuTextureUndestroy(m_backing.get());
}

void TextureImpl::setLabelInternal(const String& label)
{
    wgpuTextureSetLabel(m_backing.get(), label.utf8().data());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
