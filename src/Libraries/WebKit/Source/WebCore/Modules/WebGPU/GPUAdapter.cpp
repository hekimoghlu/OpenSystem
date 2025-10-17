/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#include "GPUAdapter.h"

#include "Exception.h"
#include "JSDOMPromiseDeferred.h"
#include "JSGPUAdapterInfo.h"
#include "JSGPUDevice.h"

#include <wtf/HashSet.h>
#include <wtf/HashTraits.h>
#include <wtf/SortedArrayMap.h>

namespace WebCore {

String GPUAdapter::name() const
{
    return m_backing->name();
}

Ref<GPUSupportedFeatures> GPUAdapter::features() const
{
    return GPUSupportedFeatures::create(WebGPU::SupportedFeatures::clone(m_backing->features()));
}

Ref<GPUSupportedLimits> GPUAdapter::limits() const
{
    return GPUSupportedLimits::create(WebGPU::SupportedLimits::clone(m_backing->limits()));
}

bool GPUAdapter::isFallbackAdapter() const
{
    return m_backing->isFallbackAdapter();
}

static WebGPU::DeviceDescriptor convertToBacking(const std::optional<GPUDeviceDescriptor>& options)
{
    if (!options)
        return { };

    return options->convertToBacking();
}

static GPUFeatureName convertFeatureNameToEnum(const String& stringValue)
{
    static constexpr std::pair<ComparableASCIILiteral, GPUFeatureName> mappings[] = {
        { "bgra8unorm-storage"_s, GPUFeatureName::Bgra8unormStorage },
        { "clip-distances"_s, GPUFeatureName::ClipDistances },
        { "depth-clip-control"_s, GPUFeatureName::DepthClipControl },
        { "depth32float-stencil8"_s, GPUFeatureName::Depth32floatStencil8 },
        { "dual-source-blending"_s, GPUFeatureName::DualSourceBlending },
        { "float16-renderable"_s, GPUFeatureName::Float16Renderable },
        { "float32-blendable"_s, GPUFeatureName::Float32Blendable },
        { "float32-filterable"_s, GPUFeatureName::Float32Filterable },
        { "float32-renderable"_s, GPUFeatureName::Float32Renderable },
        { "indirect-first-instance"_s, GPUFeatureName::IndirectFirstInstance },
        { "rg11b10ufloat-renderable"_s, GPUFeatureName::Rg11b10ufloatRenderable },
        { "shader-f16"_s, GPUFeatureName::ShaderF16 },
        { "texture-compression-astc"_s, GPUFeatureName::TextureCompressionAstc },
        { "texture-compression-astc-sliced-3d"_s, GPUFeatureName::TextureCompressionAstcSliced3d },
        { "texture-compression-bc"_s, GPUFeatureName::TextureCompressionBc },
        { "texture-compression-bc-sliced-3d"_s, GPUFeatureName::TextureCompressionBcSliced3d },
        { "texture-compression-etc2"_s, GPUFeatureName::TextureCompressionEtc2 },
        { "timestamp-query"_s, GPUFeatureName::TimestampQuery },
    };
    static constexpr SortedArrayMap enumerationMapping { mappings };
    if (auto* enumerationValue = enumerationMapping.tryGet(stringValue); LIKELY(enumerationValue))
        return *enumerationValue;

    RELEASE_ASSERT_NOT_REACHED();
}

static bool isSubset(const Vector<GPUFeatureName>& expectedSubset, const Vector<String>& expectedSuperset)
{
    HashSet<uint32_t, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> expectedSupersetHashSet;
    for (auto& featureName : expectedSuperset)
        expectedSupersetHashSet.add(static_cast<uint32_t>(convertFeatureNameToEnum(featureName)));

    for (auto& featureName : expectedSubset) {
        if (!expectedSupersetHashSet.contains(static_cast<uint32_t>(featureName)))
            return false;
    }

    return true;
}

void GPUAdapter::requestDevice(ScriptExecutionContext& scriptExecutionContext, const std::optional<GPUDeviceDescriptor>& deviceDescriptor, RequestDevicePromise&& promise)
{
    auto& existingFeatures = m_backing->features().features();
    if (deviceDescriptor && !isSubset(deviceDescriptor->requiredFeatures, existingFeatures)) {
        promise.reject(Exception(ExceptionCode::TypeError));
        return;
    }

    m_backing->requestDevice(convertToBacking(deviceDescriptor), [deviceDescriptor, promise = WTFMove(promise), scriptExecutionContextRef = Ref { scriptExecutionContext }](RefPtr<WebGPU::Device>&& device) mutable {
        if (!device.get())
            promise.reject(Exception(ExceptionCode::OperationError));
        else {
            auto queueLabel = deviceDescriptor->defaultQueue.label;
            Ref<GPUDevice> gpuDevice = GPUDevice::create(scriptExecutionContextRef.ptr(), device.releaseNonNull(), deviceDescriptor ? WTFMove(queueLabel) : ""_s);
            gpuDevice->suspendIfNeeded();
            promise.resolve(WTFMove(gpuDevice));
        }
    });
}

Ref<GPUAdapterInfo> GPUAdapter::info()
{
    return GPUAdapterInfo::create(name());
}

}
