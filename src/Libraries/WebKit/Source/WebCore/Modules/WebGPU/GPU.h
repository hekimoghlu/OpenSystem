/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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

#include "GPUAdapter.h"
#include "GPURequestAdapterOptions.h"
#include "GPUTextureFormat.h"
#include "JSDOMPromiseDeferredForward.h"
#include "WebGPU.h"
#include <optional>
#include <wtf/Deque.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class GPUCompositorIntegration;
class GPUPresentationContext;
struct GPUPresentationContextDescriptor;
class GraphicsContext;
class NativeImage;
class WGSLLanguageFeatures;

class GPU : public RefCounted<GPU> {
public:
    static Ref<GPU> create(Ref<WebGPU::GPU>&& backing)
    {
        return adoptRef(*new GPU(WTFMove(backing)));
    }
    ~GPU();

    using RequestAdapterPromise = DOMPromiseDeferred<IDLNullable<IDLInterface<GPUAdapter>>>;
    void requestAdapter(const std::optional<GPURequestAdapterOptions>&, RequestAdapterPromise&&);

    GPUTextureFormat getPreferredCanvasFormat() const;
    Ref<WGSLLanguageFeatures> wgslLanguageFeatures() const;

    RefPtr<GPUPresentationContext> createPresentationContext(const GPUPresentationContextDescriptor&);

    RefPtr<GPUCompositorIntegration> createCompositorIntegration();

    void paintToCanvas(NativeImage&, const IntSize&, GraphicsContext&);
private:
    GPU(Ref<WebGPU::GPU>&&);

    struct PendingRequestAdapterArguments;
    Ref<WebGPU::GPU> m_backing;
    Ref<WGSLLanguageFeatures> m_wgslLanguageFeatures;
};

}
