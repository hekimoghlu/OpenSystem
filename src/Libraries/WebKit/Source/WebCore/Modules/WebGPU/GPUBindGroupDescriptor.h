/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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

#include "GPUBindGroupEntry.h"
#include "GPUBindGroupLayout.h"
#include "GPUObjectDescriptorBase.h"
#include "WebGPUBindGroupDescriptor.h"
#include <wtf/Vector.h>

namespace WebCore {

struct GPUBindGroupDescriptor : public GPUObjectDescriptorBase {
    WebGPU::BindGroupDescriptor convertToBacking() const
    {
        ASSERT(layout);
        return {
            { label },
            layout->backing(),
            entries.map([](auto& bindGroupEntry) {
                return bindGroupEntry.convertToBacking();
            }),
        };
    }

    const RefPtr<GPUExternalTexture>* externalTextureMatches(Vector<GPUBindGroupEntry>& comparisonEntries, bool& hasExternalTexture) const
    {
        bool matched = true;
        hasExternalTexture = false;
        auto entriesSize = entries.size();
        if (entriesSize != comparisonEntries.size())
            matched = false;

        const RefPtr<GPUExternalTexture>* result = nullptr;
        for (size_t i = 0; i < entriesSize; ++i) {
            auto& entry = entries[i];
            if (matched && !GPUBindGroupEntry::equal(entry, comparisonEntries[i]))
                matched = false;

            auto externalTexture = entry.externalTexture();
            if (!result)
                result = externalTexture;
            else if (externalTexture)
                return nullptr;
            if (result)
                hasExternalTexture = true;
        }

        return matched ? result : nullptr;
    }

    WeakPtr<GPUBindGroupLayout> layout;
    Vector<GPUBindGroupEntry> entries;
};

}
