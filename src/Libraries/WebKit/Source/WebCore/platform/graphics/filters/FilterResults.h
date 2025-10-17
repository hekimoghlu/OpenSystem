/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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

#include "FilterEffect.h"
#include "FilterImageVector.h"
#include "ImageBufferAllocator.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FilterEffect;
class FilterImage;

class FilterResults {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(FilterResults, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT FilterResults(std::unique_ptr<ImageBufferAllocator>&& = nullptr);

    ImageBufferAllocator& allocator() const { return *m_allocator; }

    FilterImage* effectResult(FilterEffect&) const;
    void setEffectResult(FilterEffect&, const FilterImageVector& inputs, Ref<FilterImage>&& result);
    void clearEffectResult(FilterEffect&);

private:
    size_t memoryCost() const;
    bool canCacheResult(const FilterImage&) const;

    UncheckedKeyHashMap<Ref<FilterEffect>, Ref<FilterImage>> m_results;

    // The value is a list of FilterEffects, whose FilterImages depend on the key FilterImage.
    using FilterEffectSet = UncheckedKeyHashSet<Ref<FilterEffect>>;
    UncheckedKeyHashMap<Ref<FilterImage>, FilterEffectSet> m_resultReferences;

    std::unique_ptr<ImageBufferAllocator> m_allocator;
};

using FilterResultsCreator = Function<std::unique_ptr<FilterResults>()>;

} // namespace WebCore
