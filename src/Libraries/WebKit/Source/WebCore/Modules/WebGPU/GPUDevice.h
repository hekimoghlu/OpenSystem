/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "GPUBindGroupEntry.h"
#include "GPUComputePipeline.h"
#include "GPUDeviceLostInfo.h"
#include "GPUError.h"
#include "GPUErrorFilter.h"
#include "GPURenderPipeline.h"
#include "GPUQueue.h"
#include "HTMLVideoElement.h"
#include "JSDOMPromiseDeferredForward.h"
#include "ScriptExecutionContext.h"
#include "WebGPUDevice.h"
#include <optional>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUBindGroup;
struct GPUBindGroupDescriptor;
class GPUBindGroupLayout;
struct GPUBindGroupLayoutDescriptor;
class GPUBuffer;
struct GPUBufferDescriptor;
class GPUCommandEncoder;
struct GPUCommandEncoderDescriptor;
class GPUComputePipeline;
struct GPUComputePipelineDescriptor;
class GPUExternalTexture;
struct GPUExternalTextureDescriptor;
class GPURenderPipeline;
struct GPURenderPipelineDescriptor;
class GPUPipelineLayout;
struct GPUPipelineLayoutDescriptor;
class GPUPresentationContext;
class GPUQuerySet;
struct GPUQuerySetDescriptor;
class GPURenderBundleEncoder;
struct GPURenderBundleEncoderDescriptor;
class GPURenderPipeline;
struct GPURenderPipelineDescriptor;
class GPUSampler;
struct GPUSamplerDescriptor;
class GPUShaderModule;
struct GPUShaderModuleDescriptor;
class GPUSupportedFeatures;
class GPUSupportedLimits;
class GPUTexture;
struct GPUTextureDescriptor;
class WebXRSession;
class XRGPUBinding;

namespace WebGPU {
class XRBinding;
}

class GPUDevice : public RefCounted<GPUDevice>, public ActiveDOMObject, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(GPUDevice);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<GPUDevice> create(ScriptExecutionContext* scriptExecutionContext, Ref<WebGPU::Device>&& backing, String&& queueLabel)
    {
        return adoptRef(*new GPUDevice(scriptExecutionContext, WTFMove(backing), WTFMove(queueLabel)));
    }

    virtual ~GPUDevice();

    String label() const;
    void setLabel(String&&);

    Ref<GPUSupportedFeatures> features() const;
    Ref<GPUSupportedLimits> limits() const;

    Ref<GPUQueue> queue() const;

    void destroy(ScriptExecutionContext&);

    RefPtr<WebGPU::XRBinding> createXRBinding(const WebXRSession&);
    ExceptionOr<Ref<GPUBuffer>> createBuffer(const GPUBufferDescriptor&);
    ExceptionOr<Ref<GPUTexture>> createTexture(const GPUTextureDescriptor&);
    bool isSupportedFormat(GPUTextureFormat) const;
    ExceptionOr<Ref<GPUSampler>> createSampler(const std::optional<GPUSamplerDescriptor>&);
    ExceptionOr<Ref<GPUExternalTexture>> importExternalTexture(const GPUExternalTextureDescriptor&);

    ExceptionOr<Ref<GPUBindGroupLayout>> createBindGroupLayout(const GPUBindGroupLayoutDescriptor&);
    ExceptionOr<Ref<GPUPipelineLayout>> createPipelineLayout(const GPUPipelineLayoutDescriptor&);
    ExceptionOr<Ref<GPUBindGroup>> createBindGroup(const GPUBindGroupDescriptor&);

    ExceptionOr<Ref<GPUShaderModule>> createShaderModule(const GPUShaderModuleDescriptor&);
    ExceptionOr<Ref<GPUComputePipeline>> createComputePipeline(const GPUComputePipelineDescriptor&);
    ExceptionOr<Ref<GPURenderPipeline>> createRenderPipeline(const GPURenderPipelineDescriptor&);
    using CreateComputePipelineAsyncPromise = DOMPromiseDeferred<IDLInterface<GPUComputePipeline>>;
    void createComputePipelineAsync(const GPUComputePipelineDescriptor&, CreateComputePipelineAsyncPromise&&);
    using CreateRenderPipelineAsyncPromise = DOMPromiseDeferred<IDLInterface<GPURenderPipeline>>;
    ExceptionOr<void> createRenderPipelineAsync(const GPURenderPipelineDescriptor&, CreateRenderPipelineAsyncPromise&&);

    ExceptionOr<Ref<GPUCommandEncoder>> createCommandEncoder(const std::optional<GPUCommandEncoderDescriptor>&);
    ExceptionOr<Ref<GPURenderBundleEncoder>> createRenderBundleEncoder(const GPURenderBundleEncoderDescriptor&);

    ExceptionOr<Ref<GPUQuerySet>> createQuerySet(const GPUQuerySetDescriptor&);

    void pushErrorScope(GPUErrorFilter);
    using ErrorScopePromise = DOMPromiseDeferred<IDLNullable<IDLUnion<IDLInterface<GPUOutOfMemoryError>, IDLInterface<GPUValidationError>, IDLInterface<GPUInternalError>>>>;
    void popErrorScope(ErrorScopePromise&&);

    bool addEventListener(const AtomString& eventType, Ref<EventListener>&&, const AddEventListenerOptions&) override;

    using LostPromise = DOMPromiseProxy<IDLInterface<GPUDeviceLostInfo>>;
    LostPromise& lost();

    WebGPU::Device& backing() { return m_backing; }
    const WebGPU::Device& backing() const { return m_backing; }
    void removeBufferToUnmap(GPUBuffer&);
    void addBufferToUnmap(GPUBuffer&);

#if ENABLE(VIDEO)
    WeakPtr<GPUExternalTexture> takeExternalTextureForVideoElement(const HTMLVideoElement&);
#endif

private:
    GPUDevice(ScriptExecutionContext*, Ref<WebGPU::Device>&&, String&& queueLabel);

    // FIXME: We probably need to override more methods to make this work properly.
    RefPtr<GPUPipelineLayout> createAutoPipelineLayout();

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::GPUDevice; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    UniqueRef<LostPromise> m_lostPromise;
    Ref<WebGPU::Device> m_backing;
    Ref<GPUQueue> m_queue;
    RefPtr<GPUPipelineLayout> m_autoPipelineLayout;
    WeakHashSet<GPUBuffer> m_buffersToUnmap;

#if ENABLE(VIDEO)
    GPUExternalTexture* externalTextureForDescriptor(const GPUExternalTextureDescriptor&);

    WeakHashMap<HTMLVideoElement, WeakPtr<GPUExternalTexture>> m_videoElementToExternalTextureMap;
    std::pair<RefPtr<HTMLVideoElement>, RefPtr<GPUExternalTexture>> m_previouslyImportedExternalTexture;
    std::pair<Vector<GPUBindGroupEntry>, RefPtr<GPUBindGroup>> m_lastCreatedExternalTextureBindGroup;
#endif

    bool m_waitingForDeviceLostPromise { false };
};

}
