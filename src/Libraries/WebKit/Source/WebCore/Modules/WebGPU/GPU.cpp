/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#include "GPU.h"

#include "GPUPresentationContext.h"
#include "GPUPresentationContextDescriptor.h"
#include "JSDOMPromiseDeferred.h"
#include "JSGPUAdapter.h"
#include "JSWGSLLanguageFeatures.h"
#include "WGSLLanguageFeatures.h"

namespace WebCore {

GPU::GPU(Ref<WebGPU::GPU>&& backing)
    : m_backing(WTFMove(backing))
    , m_wgslLanguageFeatures(WGSLLanguageFeatures::create())
{
}

GPU::~GPU() = default;

static WebGPU::RequestAdapterOptions convertToBacking(const std::optional<GPURequestAdapterOptions>& options)
{
    if (!options)
        return { std::nullopt, false };

    return options->convertToBacking();
}

struct GPU::PendingRequestAdapterArguments {
    std::optional<GPURequestAdapterOptions> options;
    RequestAdapterPromise promise;
};

void GPU::requestAdapter(const std::optional<GPURequestAdapterOptions>& options, RequestAdapterPromise&& promise)
{
    m_backing->requestAdapter(convertToBacking(options), [promise = WTFMove(promise)](RefPtr<WebGPU::Adapter>&& adapter) mutable {
        if (!adapter) {
            promise.resolve(nullptr);
            return;
        }
        promise.resolve(GPUAdapter::create(adapter.releaseNonNull()).ptr());
    });
}

GPUTextureFormat GPU::getPreferredCanvasFormat() const
{
    return GPUTextureFormat::Bgra8unorm;
}

Ref<WGSLLanguageFeatures> GPU::wgslLanguageFeatures() const
{
    return m_wgslLanguageFeatures;
}

RefPtr<GPUPresentationContext> GPU::createPresentationContext(const GPUPresentationContextDescriptor& presentationContextDescriptor)
{
    RefPtr context = m_backing->createPresentationContext(presentationContextDescriptor.convertToBacking());
    if (!context)
        return nullptr;
    return GPUPresentationContext::create(context.releaseNonNull());
}

RefPtr<GPUCompositorIntegration> GPU::createCompositorIntegration()
{
    RefPtr integration = m_backing->createCompositorIntegration();
    if (!integration)
        return nullptr;
    return GPUCompositorIntegration::create(integration.releaseNonNull());
}

} // namespace WebCore
