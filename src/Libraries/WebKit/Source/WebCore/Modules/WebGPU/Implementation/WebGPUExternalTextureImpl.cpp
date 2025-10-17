/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#include "WebGPUExternalTextureImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include "WebGPUExternalTextureDescriptor.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ExternalTextureImpl);

ExternalTextureImpl::ExternalTextureImpl(WebGPUPtr<WGPUExternalTexture>&& externalTexture, const ExternalTextureDescriptor& descriptor, ConvertToBackingContext& convertToBackingContext)
    : m_convertToBackingContext(convertToBackingContext)
    , m_backing(WTFMove(externalTexture))
    , m_colorSpace(descriptor.colorSpace)
{
    UNUSED_VARIABLE(m_colorSpace);
}

ExternalTextureImpl::~ExternalTextureImpl() = default;

void ExternalTextureImpl::setLabelInternal(const String&)
{
    // FIXME: Implement this.
}

void ExternalTextureImpl::destroy()
{
    wgpuExternalTextureDestroy(m_backing.get());
}

void ExternalTextureImpl::undestroy()
{
    wgpuExternalTextureUndestroy(m_backing.get());
}

void ExternalTextureImpl::updateExternalTexture(CVPixelBufferRef pixelBuffer)
{
    wgpuExternalTextureUpdate(m_backing.get(), pixelBuffer);
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
