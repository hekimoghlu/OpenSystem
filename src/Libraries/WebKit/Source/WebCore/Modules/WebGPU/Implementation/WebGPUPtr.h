/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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

#include <WebGPU/WebGPU.h>
#include <WebGPU/WebGPUExt.h>
#include <type_traits>
#include <wtf/RefPtr.h>

namespace WebCore::WebGPU {

template<typename WGPUT>
struct WebGPUPtrTraits {
    using StorageType = WGPUT;

    template<typename U>
    static ALWAYS_INLINE StorageType exchange(StorageType& ptr, U&& newValue) { return std::exchange(ptr, newValue); }

    static ALWAYS_INLINE void swap(StorageType& a, StorageType& b) { std::swap(a, b); }
    static ALWAYS_INLINE WGPUT unwrap(const StorageType& ptr) { return ptr; }

    static StorageType hashTableDeletedValue() { return std::bit_cast<StorageType>(static_cast<uintptr_t>(-1)); }
    static ALWAYS_INLINE bool isHashTableDeletedValue(const StorageType& ptr) { return ptr == hashTableDeletedValue(); }
};

template <typename T, void (*reference)(T), void(*release)(T)> struct BaseWebGPURefDerefTraits {
    static ALWAYS_INLINE T refIfNotNull(T t)
    {
        if (LIKELY(t))
            reference(t);
        return t;
    }

    static ALWAYS_INLINE void derefIfNotNull(T t)
    {
        if (LIKELY(t))
            release(t);
    }
};

template <typename> struct WebGPURefDerefTraits;

template <> struct WebGPURefDerefTraits<WGPUAdapter> : public BaseWebGPURefDerefTraits<WGPUAdapter, wgpuAdapterReference, wgpuAdapterRelease> { };
template <> struct WebGPURefDerefTraits<WGPUBindGroup> : public BaseWebGPURefDerefTraits<WGPUBindGroup, wgpuBindGroupReference, wgpuBindGroupRelease> { };
template <> struct WebGPURefDerefTraits<WGPUBindGroupLayout> : public BaseWebGPURefDerefTraits<WGPUBindGroupLayout, wgpuBindGroupLayoutReference, wgpuBindGroupLayoutRelease> { };
template <> struct WebGPURefDerefTraits<WGPUBuffer> : public BaseWebGPURefDerefTraits<WGPUBuffer, wgpuBufferReference, wgpuBufferRelease> { };
template <> struct WebGPURefDerefTraits<WGPUCommandBuffer> : public BaseWebGPURefDerefTraits<WGPUCommandBuffer, wgpuCommandBufferReference, wgpuCommandBufferRelease> { };
template <> struct WebGPURefDerefTraits<WGPUCommandEncoder> : public BaseWebGPURefDerefTraits<WGPUCommandEncoder, wgpuCommandEncoderReference, wgpuCommandEncoderRelease> { };
template <> struct WebGPURefDerefTraits<WGPUComputePassEncoder> : public BaseWebGPURefDerefTraits<WGPUComputePassEncoder, wgpuComputePassEncoderReference, wgpuComputePassEncoderRelease> { };
template <> struct WebGPURefDerefTraits<WGPUComputePipeline> : public BaseWebGPURefDerefTraits<WGPUComputePipeline, wgpuComputePipelineReference, wgpuComputePipelineRelease> { };
template <> struct WebGPURefDerefTraits<WGPUDevice> : public BaseWebGPURefDerefTraits<WGPUDevice, wgpuDeviceReference, wgpuDeviceRelease> { };
template <> struct WebGPURefDerefTraits<WGPUInstance> : public BaseWebGPURefDerefTraits<WGPUInstance, wgpuInstanceReference, wgpuInstanceRelease> { };
template <> struct WebGPURefDerefTraits<WGPUPipelineLayout> : public BaseWebGPURefDerefTraits<WGPUPipelineLayout, wgpuPipelineLayoutReference, wgpuPipelineLayoutRelease> { };
template <> struct WebGPURefDerefTraits<WGPUQuerySet> : public BaseWebGPURefDerefTraits<WGPUQuerySet, wgpuQuerySetReference, wgpuQuerySetRelease> { };
template <> struct WebGPURefDerefTraits<WGPUQueue> : public BaseWebGPURefDerefTraits<WGPUQueue, wgpuQueueReference, wgpuQueueRelease> { };
template <> struct WebGPURefDerefTraits<WGPURenderBundle> : public BaseWebGPURefDerefTraits<WGPURenderBundle, wgpuRenderBundleReference, wgpuRenderBundleRelease> { };
template <> struct WebGPURefDerefTraits<WGPURenderBundleEncoder> : public BaseWebGPURefDerefTraits<WGPURenderBundleEncoder, wgpuRenderBundleEncoderReference, wgpuRenderBundleEncoderRelease> { };
template <> struct WebGPURefDerefTraits<WGPURenderPassEncoder> : public BaseWebGPURefDerefTraits<WGPURenderPassEncoder, wgpuRenderPassEncoderReference, wgpuRenderPassEncoderRelease> { };
template <> struct WebGPURefDerefTraits<WGPURenderPipeline> : public BaseWebGPURefDerefTraits<WGPURenderPipeline, wgpuRenderPipelineReference, wgpuRenderPipelineRelease> { };
template <> struct WebGPURefDerefTraits<WGPUSampler> : public BaseWebGPURefDerefTraits<WGPUSampler, wgpuSamplerReference, wgpuSamplerRelease> { };
template <> struct WebGPURefDerefTraits<WGPUShaderModule> : public BaseWebGPURefDerefTraits<WGPUShaderModule, wgpuShaderModuleReference, wgpuShaderModuleRelease> { };
template <> struct WebGPURefDerefTraits<WGPUSurface> : public BaseWebGPURefDerefTraits<WGPUSurface, wgpuSurfaceReference, wgpuSurfaceRelease> { };
template <> struct WebGPURefDerefTraits<WGPUSwapChain> : public BaseWebGPURefDerefTraits<WGPUSwapChain, wgpuSwapChainReference, wgpuSwapChainRelease> { };
template <> struct WebGPURefDerefTraits<WGPUTexture> : public BaseWebGPURefDerefTraits<WGPUTexture, wgpuTextureReference, wgpuTextureRelease> { };
template <> struct WebGPURefDerefTraits<WGPUTextureView> : public BaseWebGPURefDerefTraits<WGPUTextureView, wgpuTextureViewReference, wgpuTextureViewRelease> { };
template <> struct WebGPURefDerefTraits<WGPUExternalTexture> : public BaseWebGPURefDerefTraits<WGPUExternalTexture, wgpuExternalTextureReference, wgpuExternalTextureRelease> { };
template <> struct WebGPURefDerefTraits<WGPUXRBinding> : public BaseWebGPURefDerefTraits<WGPUXRBinding, wgpuXRBindingReference, wgpuXRBindingRelease> { };
template <> struct WebGPURefDerefTraits<WGPUXRProjectionLayer> : public BaseWebGPURefDerefTraits<WGPUXRProjectionLayer, wgpuXRProjectionLayerReference, wgpuXRProjectionLayerRelease> { };
template <> struct WebGPURefDerefTraits<WGPUXRSubImage> : public BaseWebGPURefDerefTraits<WGPUXRSubImage, wgpuXRSubImageReference, wgpuXRSubImageRelease> { };
template <> struct WebGPURefDerefTraits<WGPUXRView> : public BaseWebGPURefDerefTraits<WGPUXRView, wgpuXRViewReference, wgpuXRViewRelease> { };

template <typename T> using WebGPUPtr = RefPtr<std::remove_pointer_t<T>, WebGPUPtrTraits<T>, WebGPURefDerefTraits<T>>;

template <typename T> inline WebGPUPtr<T> adoptWebGPU(T t)
{
    return adoptRef<std::remove_pointer_t<T>, WebGPUPtrTraits<T>, WebGPURefDerefTraits<T>>(t);
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
