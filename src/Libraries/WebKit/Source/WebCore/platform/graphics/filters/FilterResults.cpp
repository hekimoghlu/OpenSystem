/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#include "FilterResults.h"

#include "ImageBuffer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FilterResults);

FilterResults::FilterResults(std::unique_ptr<ImageBufferAllocator>&& allocator)
    : m_allocator(allocator ? WTFMove(allocator) : makeUnique<ImageBufferAllocator>())
{
}

FilterImage* FilterResults::effectResult(FilterEffect& effect) const
{
    return m_results.get(effect);
}

size_t FilterResults::memoryCost() const
{
    CheckedSize memoryCost;

    for (auto& result : m_results.values())
        memoryCost += result->memoryCost();

    return memoryCost;
}

bool FilterResults::canCacheResult(const FilterImage& result) const
{
    static constexpr size_t maxAllowedMemoryCost = 100 * MB;
    CheckedSize totalMemoryCost = memoryCost();

    totalMemoryCost += result.memoryCost();
    if (totalMemoryCost.hasOverflowed())
        return false;

    return totalMemoryCost <= maxAllowedMemoryCost;
}

void FilterResults::setEffectResult(FilterEffect& effect, const FilterImageVector& inputs, Ref<FilterImage>&& result)
{
    if (!canCacheResult(result))
        return;

    m_results.set({ effect }, WTFMove(result));

    for (auto& input : inputs)
        m_resultReferences.add(input, FilterEffectSet()).iterator->value.add(effect);
}

void FilterResults::clearEffectResult(FilterEffect& effect)
{
    auto iterator = m_results.find(effect);
    if (iterator == m_results.end())
        return;

    auto result = iterator->value;
    m_results.remove(iterator);

    for (auto& reference : m_resultReferences.get(result))
        clearEffectResult(reference);

    m_resultReferences.remove(result);
}

} // namespace WebCore
