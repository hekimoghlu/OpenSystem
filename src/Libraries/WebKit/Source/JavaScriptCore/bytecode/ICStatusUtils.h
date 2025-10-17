/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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

#include "BytecodeIndex.h"
#include "CacheableIdentifier.h"
#include "ExitFlag.h"

namespace JSC {

class CodeBlock;
class StructureSet;

template<typename VariantVectorType, typename VariantType>
bool appendICStatusVariant(VariantVectorType& variants, const VariantType& variant)
{
    // Attempt to merge this variant with an already existing variant.
    for (unsigned i = 0; i < variants.size(); ++i) {
        VariantType& mergedVariant = variants[i];
        if (mergedVariant.attemptToMerge(variant)) {
            for (unsigned j = 0; j < variants.size(); ++j) {
                if (i == j)
                    continue;
                if (variants[j].overlaps(mergedVariant))
                    return false;
            }
            return true;
        }
    }
    
    // Make sure there is no overlap. We should have pruned out opportunities for
    // overlap but it's possible that an inline cache got into a weird state. We are
    // defensive and bail if we detect crazy.
    for (unsigned i = 0; i < variants.size(); ++i) {
        if (variants[i].overlaps(variant))
            return false;
    }
    
    variants.append(variant);
    return true;
}

template<typename VariantVectorType>
void filterICStatusVariants(VariantVectorType& variants, const StructureSet& set)
{
    variants.removeAllMatching(
        [&] (auto& variant) -> bool {
            variant.structureSet().filter(set);
            return variant.structureSet().isEmpty();
        });
}

template<typename VariantVectorType>
CacheableIdentifier singleIdentifierForICStatus(VariantVectorType& variants)
{
    if (variants.isEmpty())
        return nullptr;

    CacheableIdentifier result = variants.first().identifier();
    if (!result)
        return nullptr;

    for (size_t i = 1; i < variants.size(); ++i) {
        CacheableIdentifier identifier = variants[i].identifier();
        if (!identifier || identifier != result)
            return nullptr;
    }

    return result;
}

ExitFlag hasBadCacheExitSite(CodeBlock* profiledBlock, BytecodeIndex);

} // namespace JSC

