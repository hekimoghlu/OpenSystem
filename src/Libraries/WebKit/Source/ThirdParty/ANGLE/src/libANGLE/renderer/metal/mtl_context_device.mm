/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

//
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_context_device.mm:
//      Implementation of Metal framework's MTLDevice wrapper per context.
//

#include "libANGLE/renderer/metal/mtl_context_device.h"
#if ANGLE_USE_METAL_OWNERSHIP_IDENTITY
#    include "libANGLE/renderer/metal/mtl_resource_spi.h"
#endif

namespace rx
{
namespace mtl
{

ContextDevice::ContextDevice(GLint ownershipIdentity)
{
#if ANGLE_USE_METAL_OWNERSHIP_IDENTITY
    mOwnershipIdentity = static_cast<task_id_token_t>(ownershipIdentity);
    if (mOwnershipIdentity != TASK_ID_TOKEN_NULL)
    {
        kern_return_t kr =
            mach_port_mod_refs(mach_task_self(), mOwnershipIdentity, MACH_PORT_RIGHT_SEND, 1);
        if (ANGLE_UNLIKELY(kr != KERN_SUCCESS))
        {
            ERR() << "mach_port_mod_refs failed with: %s (%x)" << mach_error_string(kr) << kr;
            ASSERT(false);
        }
    }
#endif
}

ContextDevice::~ContextDevice()
{
#if ANGLE_USE_METAL_OWNERSHIP_IDENTITY
    if (mOwnershipIdentity != TASK_ID_TOKEN_NULL)
    {
        kern_return_t kr =
            mach_port_mod_refs(mach_task_self(), mOwnershipIdentity, MACH_PORT_RIGHT_SEND, -1);
        if (ANGLE_UNLIKELY(kr != KERN_SUCCESS))
        {
            ERR() << "mach_port_mod_refs failed with: %s (%x)" << mach_error_string(kr) << kr;
            ASSERT(false);
        }
    }
#endif
}

AutoObjCPtr<id<MTLSamplerState>> ContextDevice::newSamplerStateWithDescriptor(
    MTLSamplerDescriptor *descriptor) const
{
    return adoptObjCObj([get() newSamplerStateWithDescriptor:descriptor]);
}

AutoObjCPtr<id<MTLTexture>> ContextDevice::newTextureWithDescriptor(
    MTLTextureDescriptor *descriptor) const
{

    auto resource = adoptObjCObj([get() newTextureWithDescriptor:descriptor]);
    setOwnerWithIdentity(resource);
    return resource;
}

AutoObjCPtr<id<MTLTexture>> ContextDevice::newTextureWithDescriptor(
    MTLTextureDescriptor *descriptor,
    IOSurfaceRef iosurface,
    NSUInteger plane) const
{
    return adoptObjCObj([get() newTextureWithDescriptor:descriptor
                                              iosurface:iosurface
                                                  plane:plane]);
}

AutoObjCPtr<id<MTLBuffer>> ContextDevice::newBufferWithLength(NSUInteger length,
                                                              MTLResourceOptions options) const
{
    auto resource = adoptObjCObj([get() newBufferWithLength:length options:options]);
    setOwnerWithIdentity(resource);
    return resource;
}

AutoObjCPtr<id<MTLBuffer>> ContextDevice::newBufferWithBytes(const void *pointer,
                                                             NSUInteger length,
                                                             MTLResourceOptions options) const
{
    auto resource = adoptObjCObj([get() newBufferWithBytes:pointer length:length options:options]);
    setOwnerWithIdentity(resource);
    return resource;
}

AutoObjCPtr<id<MTLComputePipelineState>> ContextDevice::newComputePipelineStateWithFunction(
    id<MTLFunction> computeFunction,
    __autoreleasing NSError **error) const
{
    return adoptObjCObj([get() newComputePipelineStateWithFunction:computeFunction error:error]);
}

AutoObjCPtr<id<MTLRenderPipelineState>> ContextDevice::newRenderPipelineStateWithDescriptor(
    MTLRenderPipelineDescriptor *descriptor,
    __autoreleasing NSError **error) const
{
    return adoptObjCObj([get() newRenderPipelineStateWithDescriptor:descriptor error:error]);
}

AutoObjCPtr<id<MTLDepthStencilState>> ContextDevice::newDepthStencilStateWithDescriptor(
    MTLDepthStencilDescriptor *descriptor) const
{
    return adoptObjCObj([get() newDepthStencilStateWithDescriptor:descriptor]);
}

AutoObjCPtr<id<MTLSharedEvent>> ContextDevice::newSharedEvent() const
{
    return adoptObjCObj([get() newSharedEvent]);
}

AutoObjCPtr<id<MTLEvent>> ContextDevice::newEvent() const
{
    return adoptObjCObj([get() newEvent]);
}

void ContextDevice::setOwnerWithIdentity(id<MTLResource> resource) const
{
#if ANGLE_USE_METAL_OWNERSHIP_IDENTITY
    mtl::setOwnerWithIdentity(resource, mOwnershipIdentity);
#endif
}

bool ContextDevice::hasUnifiedMemory() const
{
    return [get() hasUnifiedMemory];
}

}  // namespace mtl
}  // namespace rx
