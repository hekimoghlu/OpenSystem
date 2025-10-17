/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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

#import <wtf/FastMalloc.h>
#import <wtf/ListHashSet.h>
#import <wtf/Lock.h>
#import <wtf/Ref.h>
#import <wtf/RefCounted.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

struct WGPUSamplerImpl {
};

namespace WebGPU {

class Device;

// https://gpuweb.github.io/gpuweb/#gpusampler
class Sampler : public WGPUSamplerImpl, public RefCounted<Sampler> {
    WTF_MAKE_TZONE_ALLOCATED(Sampler);
public:
    using UniqueSamplerIdentifier = String;

    static Ref<Sampler> create(UniqueSamplerIdentifier&& samplerIdentifier, const WGPUSamplerDescriptor& descriptor, Device& device)
    {
        return adoptRef(*new Sampler(WTFMove(samplerIdentifier), descriptor, device));
    }
    static Ref<Sampler> createInvalid(Device& device)
    {
        return adoptRef(*new Sampler(device));
    }

    ~Sampler();

    void setLabel(String&&);

    bool isValid() const;

    id<MTLSamplerState> cachedSampler() const;
    id<MTLSamplerState> samplerState() const;
    const WGPUSamplerDescriptor& descriptor() const { return m_descriptor; }
    bool isComparison() const { return descriptor().compare != WGPUCompareFunction_Undefined; }
    bool isFiltering() const { return descriptor().minFilter == WGPUFilterMode_Linear || descriptor().magFilter == WGPUFilterMode_Linear || descriptor().mipmapFilter == WGPUMipmapFilterMode_Linear; }

    Device& device() const { return m_device; }

private:
    Sampler(UniqueSamplerIdentifier&&, const WGPUSamplerDescriptor&, Device&);
    Sampler(Device&);

    std::optional<UniqueSamplerIdentifier> m_samplerIdentifier;
    WGPUSamplerDescriptor m_descriptor { };

    const Ref<Device> m_device;
    // static is intentional here as the limit is per process
    static Lock samplerStateLock;
    using CachedSamplerStateContainer = HashMap<UniqueSamplerIdentifier, WeakObjCPtr<id<MTLSamplerState>>>;
    struct SamplerStateWithReferences {
        RetainPtr<id<MTLSamplerState>> samplerState;
        HashSet<const Sampler*> apiSamplerList;
    };
    using RetainedSamplerStateContainer = HashMap<UniqueSamplerIdentifier, SamplerStateWithReferences>;
    using CachedKeyContainer = ListHashSet<UniqueSamplerIdentifier>;
    static std::unique_ptr<CachedSamplerStateContainer> cachedSamplerStates WTF_GUARDED_BY_LOCK(samplerStateLock);
    static std::unique_ptr<RetainedSamplerStateContainer> retainedSamplerStates WTF_GUARDED_BY_LOCK(samplerStateLock);
    static std::unique_ptr<CachedKeyContainer> lastAccessedKeys;

    mutable __weak id<MTLSamplerState> m_cachedSamplerState { nil };
};

} // namespace WebGPU
