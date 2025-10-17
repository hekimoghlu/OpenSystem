/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#include "WebGPUBindGroupImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include "WebGPUExternalTextureImpl.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BindGroupImpl);

BindGroupImpl::BindGroupImpl(WebGPUPtr<WGPUBindGroup>&& bindGroup, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(bindGroup))
    , m_convertToBackingContext(convertToBackingContext)
{
}

BindGroupImpl::~BindGroupImpl() = default;

void BindGroupImpl::setLabelInternal(const String& label)
{
    wgpuBindGroupSetLabel(m_backing.get(), label.utf8().data());
}

bool BindGroupImpl::updateExternalTextures(ExternalTexture& externalTexture)
{
    return wgpuBindGroupUpdateExternalTextures(m_backing.get(), static_cast<const ExternalTextureImpl&>(externalTexture).backing());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
