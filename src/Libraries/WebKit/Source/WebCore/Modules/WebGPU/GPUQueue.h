/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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

#include "BufferSource.h"
#include "GPUCommandBuffer.h"
#include "GPUExtent3DDict.h"
#include "GPUImageCopyExternalImage.h"
#include "GPUImageCopyTexture.h"
#include "GPUImageCopyTextureTagged.h"
#include "GPUImageDataLayout.h"
#include "GPUIntegralTypes.h"
#include "WebGPUQueue.h"
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUBuffer;

namespace WebGPU {
class Device;
}

class GPUQueue : public RefCounted<GPUQueue> {
public:
    static Ref<GPUQueue> create(Ref<WebGPU::Queue>&& backing, WebGPU::Device& device)
    {
        return adoptRef(*new GPUQueue(WTFMove(backing), device));
    }

    String label() const;
    void setLabel(String&&);

    void submit(Vector<Ref<GPUCommandBuffer>>&&);

    using OnSubmittedWorkDonePromise = DOMPromiseDeferred<IDLNull>;
    void onSubmittedWorkDone(OnSubmittedWorkDonePromise&&);

    ExceptionOr<void> writeBuffer(
        const GPUBuffer&,
        GPUSize64 bufferOffset,
        BufferSource&& data,
        std::optional<GPUSize64> dataOffset,
        std::optional<GPUSize64>);

    void writeTexture(
        const GPUImageCopyTexture& destination,
        BufferSource&& data,
        const GPUImageDataLayout&,
        const GPUExtent3D& size);

    ExceptionOr<void> copyExternalImageToTexture(
        ScriptExecutionContext&,
        const GPUImageCopyExternalImage& source,
        const GPUImageCopyTextureTagged& destination,
        const GPUExtent3D& copySize);

    WebGPU::Queue& backing() { return m_backing; }
    const WebGPU::Queue& backing() const { return m_backing; }

private:
    GPUQueue(Ref<WebGPU::Queue>&&, WebGPU::Device&);

    Ref<WebGPU::Queue> m_backing;
    WeakPtr<WebGPU::Device> m_device;
};

}
