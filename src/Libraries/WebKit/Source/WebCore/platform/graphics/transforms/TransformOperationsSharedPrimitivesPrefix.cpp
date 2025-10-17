/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
#include "TransformOperationsSharedPrimitivesPrefix.h"

#include "TransformOperations.h"
#include <algorithm>

namespace WebCore {

void TransformOperationsSharedPrimitivesPrefix::update(const TransformOperations& operations)
{
    size_t maxIteration = operations.size();
    if (m_indexOfFirstMismatch.has_value())
        maxIteration = std::min(*m_indexOfFirstMismatch, maxIteration);

    for (size_t i = 0; i < maxIteration; ++i) {
        Ref operation = operations[i];

        // If we haven't seen an operation at this index before, we can simply use our primitive type.
        if (i >= m_primitives.size()) {
            ASSERT(i == m_primitives.size());
            m_primitives.append(operation->primitiveType());
            continue;
        }

        if (auto sharedPrimitive = operation->sharedPrimitiveType(m_primitives[i]))
            m_primitives[i] = *sharedPrimitive;
        else {
            m_indexOfFirstMismatch = i;
            m_primitives.shrink(i);
            return;
        }
    }
}

} // namespace WebCore
