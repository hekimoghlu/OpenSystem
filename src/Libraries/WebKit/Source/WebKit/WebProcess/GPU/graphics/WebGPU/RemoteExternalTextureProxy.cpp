/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#include "RemoteExternalTextureProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteExternalTextureMessages.h"
#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteExternalTextureProxy);

RemoteExternalTextureProxy::RemoteExternalTextureProxy(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
{
}

RemoteExternalTextureProxy::~RemoteExternalTextureProxy()
{
    auto sendResult = send(Messages::RemoteExternalTexture::Destruct());
    UNUSED_VARIABLE(sendResult);
}

void RemoteExternalTextureProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteExternalTexture::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

void RemoteExternalTextureProxy::destroy()
{
    auto sendResult = send(Messages::RemoteExternalTexture::Destroy());
    UNUSED_VARIABLE(sendResult);
}

void RemoteExternalTextureProxy::undestroy()
{
    auto sendResult = send(Messages::RemoteExternalTexture::Undestroy());
    UNUSED_VARIABLE(sendResult);
}

void RemoteExternalTextureProxy::updateExternalTexture(CVPixelBufferRef)
{
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
