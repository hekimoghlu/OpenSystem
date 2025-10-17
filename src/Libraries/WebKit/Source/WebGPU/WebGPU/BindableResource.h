/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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

#import <Metal/Metal.h>
#import <wtf/OptionSet.h>
#import <wtf/RefPtr.h>
#import <wtf/Vector.h>
#import <wtf/WeakPtr.h>

namespace WebGPU {

class Buffer;
class ExternalTexture;
class TextureView;

enum class BindGroupEntryUsage {
    Undefined = 0,
    Input = 1 << 0,
    Constant = 1 << 1,
    Storage = 1 << 2,
    StorageRead = 1 << 3,
    Attachment = 1 << 4,
    AttachmentRead = 1 << 5,
    ConstantTexture = 1 << 6,
    StorageTextureWriteOnly = 1 << 7,
    StorageTextureRead = 1 << 8,
    StorageTextureReadWrite = 1 << 9,
};

static constexpr auto isTextureBindGroupEntryUsage(OptionSet<BindGroupEntryUsage> usage)
{
    return usage.toRaw() >= static_cast<std::underlying_type<BindGroupEntryUsage>::type>(BindGroupEntryUsage::Attachment);
}

struct BindGroupEntryUsageData {
    OptionSet<BindGroupEntryUsage> usage { BindGroupEntryUsage::Undefined };
    uint32_t binding { 0 };
    using Resource = std::variant<RefPtr<Buffer>, RefPtr<const TextureView>, RefPtr<const ExternalTexture>>;
    Resource resource;
    uint64_t entryOffset { 0 };
    uint64_t entrySize { 0 };
    static constexpr uint32_t invalidBindingIndex = INT_MAX;
    static constexpr BindGroupEntryUsage invalidBindGroupUsage = static_cast<BindGroupEntryUsage>(std::numeric_limits<std::underlying_type<BindGroupEntryUsage>::type>::max());
};

struct BindableResources {
    Vector<id<MTLResource>> mtlResources;
    Vector<BindGroupEntryUsageData> resourceUsages;
    MTLResourceUsage usage;
    MTLRenderStages renderStages;
};

struct IndexData {
    uint64_t renderCommand { 0 };
    uint32_t minVertexCount { UINT32_MAX };
    uint32_t minInstanceCount { UINT32_MAX };
    uint64_t bufferGpuAddress { 0 };
    uint32_t indexCount { 0 };
    uint32_t instanceCount { 0 };
    uint32_t firstIndex { 0 };
    int32_t baseVertex { 0 };
    uint32_t firstInstance { 0 };
    MTLPrimitiveType primitiveType { MTLPrimitiveTypeTriangle };
};

struct IndexBufferAndIndexData {
    RefPtr<Buffer> indexBuffer;
    MTLIndexType indexType { MTLIndexTypeUInt16 };
    NSUInteger indexBufferOffsetInBytes { 0 };
    IndexData indexData;
};

} // namespace WebGPU
