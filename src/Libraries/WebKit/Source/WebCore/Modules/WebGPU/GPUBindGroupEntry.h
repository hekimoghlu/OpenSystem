/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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

#include "GPUBufferBinding.h"
#include "GPUExternalTexture.h"
#include "GPUIntegralTypes.h"
#include "GPUSampler.h"
#include "GPUTextureView.h"
#include "WebGPUBindGroupEntry.h"
#include <utility>
#include <variant>

namespace WebCore {

using GPUBindingResource = std::variant<RefPtr<GPUSampler>, RefPtr<GPUTextureView>, GPUBufferBinding, RefPtr<GPUExternalTexture>>;

inline WebGPU::BindingResource convertToBacking(const GPUBindingResource& bindingResource)
{
    return WTF::switchOn(bindingResource, [](const RefPtr<GPUSampler>& sampler) -> WebGPU::BindingResource {
        ASSERT(sampler);
        return sampler->backing();
    }, [](const RefPtr<GPUTextureView>& textureView) -> WebGPU::BindingResource {
        ASSERT(textureView);
        return textureView->backing();
    }, [](const GPUBufferBinding& bufferBinding) -> WebGPU::BindingResource {
        return bufferBinding.convertToBacking();
    }, [](const RefPtr<GPUExternalTexture>& externalTexture) -> WebGPU::BindingResource {
        ASSERT(externalTexture);
        return externalTexture->backing();
    });
}

struct GPUBindGroupEntry {
    WebGPU::BindGroupEntry convertToBacking() const
    {
        return {
            binding,
            WebCore::convertToBacking(resource),
        };
    }

    static bool equal(const GPUSampler& entry, const GPUBindingResource& otherEntry)
    {
        return WTF::switchOn(otherEntry, [&](const RefPtr<GPUSampler>& sampler) -> bool {
            return sampler.get() == &entry;
        }, [&](const RefPtr<GPUTextureView>&) -> bool {
            return false;
        }, [&](const GPUBufferBinding&) -> bool {
            return false;
        }, [&](const RefPtr<GPUExternalTexture>&) -> bool {
            return false;
        });
    }
    static bool equal(const GPUTextureView& entry, const GPUBindingResource& otherEntry)
    {
        return WTF::switchOn(otherEntry, [&](const RefPtr<GPUSampler>&) -> bool {
            return false;
        }, [&](const RefPtr<GPUTextureView>& textureView) -> bool {
            return textureView.get() == &entry;
        }, [&](const GPUBufferBinding&) -> bool {
            return false;
        }, [&](const RefPtr<GPUExternalTexture>&) -> bool {
            return false;
        });
    }
    static bool equalSizes(const std::optional<GPUSize64>& a, const std::optional<GPUSize64>& b)
    {
        return (!a && !b) || (a && b && *a == *b);
    }
    static bool equal(const GPUBufferBinding& entry, const GPUBindingResource& otherEntry)
    {
        return WTF::switchOn(otherEntry, [&](const RefPtr<GPUSampler>&) -> bool {
            return false;
        }, [&](const RefPtr<GPUTextureView>&) -> bool {
            return false;
        }, [&](const GPUBufferBinding& bufferBinding) -> bool {
            return bufferBinding.buffer.get() == entry.buffer.get() && bufferBinding.offset == entry.offset && equalSizes(bufferBinding.size, entry.size);
        }, [&](const RefPtr<GPUExternalTexture>&) -> bool {
            return false;
        });
    }
    static bool equal(const GPUExternalTexture& entry, const GPUBindingResource& otherEntry)
    {
        return WTF::switchOn(otherEntry, [&](const RefPtr<GPUSampler>&) -> bool {
            return false;
        }, [&](const RefPtr<GPUTextureView>&) -> bool {
            return false;
        }, [&](const GPUBufferBinding&) -> bool {
            return false;
        }, [&](const RefPtr<GPUExternalTexture>& externalTexture) -> bool {
            return externalTexture.get() == &entry;
        });
    }
    static bool equal(const GPUBindGroupEntry& entry, const GPUBindGroupEntry& otherEntry)
    {
        if (entry.binding != otherEntry.binding)
            return false;

        return WTF::switchOn(entry.resource, [&](const RefPtr<GPUSampler>& sampler) -> bool {
            return sampler.get() && equal(*sampler, otherEntry.resource);
        }, [&](const RefPtr<GPUTextureView>& textureView) -> bool {
            return textureView.get() && equal(*textureView, otherEntry.resource);
        }, [&](const GPUBufferBinding& bufferBinding) -> bool {
            return equal(bufferBinding, otherEntry.resource);
        }, [&](const RefPtr<GPUExternalTexture>& externalTexture) -> bool {
            return externalTexture.get() && equal(*externalTexture, otherEntry.resource);
        });
    }

    const RefPtr<GPUExternalTexture>* externalTexture() const
    {
        return std::get_if<RefPtr<GPUExternalTexture>>(&resource);
    }

    GPUIndex32 binding { 0 };
    GPUBindingResource resource;
};

}
