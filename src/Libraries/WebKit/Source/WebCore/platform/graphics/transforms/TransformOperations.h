/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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

#include "LayoutSize.h"
#include "TransformOperation.h"
#include <algorithm>
#include <wtf/ArgumentCoder.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

struct BlendingContext;

class TransformOperations {
    WTF_MAKE_TZONE_ALLOCATED(TransformOperations);
public:
    using const_iterator = Vector<Ref<TransformOperation>>::const_iterator;
    using const_reverse_iterator = Vector<Ref<TransformOperation>>::const_reverse_iterator;
    using value_type = Vector<Ref<TransformOperation>>::value_type;

    TransformOperations() = default;

    explicit TransformOperations(Ref<TransformOperation>&&);
    WEBCORE_EXPORT explicit TransformOperations(Vector<Ref<TransformOperation>>&&);

    bool operator==(const TransformOperations&) const;

    WEBCORE_EXPORT TransformOperations clone() const;
    TransformOperations selfOrCopyWithResolvedCalculatedValues(const FloatSize&) const;

    const_iterator begin() const { return m_operations.begin(); }
    const_iterator end() const { return m_operations.end(); }
    const_reverse_iterator rbegin() const { return m_operations.rbegin(); }
    const_reverse_iterator rend() const { return m_operations.rend(); }

    bool isEmpty() const { return m_operations.isEmpty(); }
    size_t size() const { return m_operations.size(); }
    const TransformOperation* at(size_t index) const { return index < m_operations.size() ? m_operations[index].ptr() : nullptr; }

    const Ref<TransformOperation>& operator[](size_t i) const { return m_operations[i]; }
    const Ref<TransformOperation>& first() const { return m_operations.first(); }
    const Ref<TransformOperation>& last() const { return m_operations.last(); }

    void apply(TransformationMatrix&, const FloatSize&, unsigned start = 0) const;

    // Return true if any of the operation types are 3D operation types (even if the
    // values describe affine transforms)
    bool has3DOperation() const;
    bool isRepresentableIn2D() const;
    bool affectedByTransformOrigin() const;

    template<TransformOperation::Type operationType>
    bool hasTransformOfType() const;

    bool isInvertible(const LayoutSize&) const;

    bool containsNonInvertibleMatrix(const LayoutSize&) const;
    bool shouldFallBackToDiscreteAnimation(const TransformOperations&, const LayoutSize&) const;

    Ref<TransformOperation> createBlendedMatrixOperationFromOperationsSuffix(const TransformOperations& from, unsigned start, const BlendingContext&, const LayoutSize& referenceBoxSize) const;
    TransformOperations blend(const TransformOperations& from, const BlendingContext&, const LayoutSize&, std::optional<unsigned> prefixLength = std::nullopt) const;

private:
    friend struct IPC::ArgumentCoder<TransformOperations, void>;
    friend WTF::TextStream& operator<<(WTF::TextStream&, const TransformOperations&);

    Vector<Ref<TransformOperation>> m_operations;
};

template<TransformOperation::Type operationType>
bool TransformOperations::hasTransformOfType() const
{
    return std::ranges::any_of(m_operations, [](auto& op) { return op->type() == operationType; });
}

WTF::TextStream& operator<<(WTF::TextStream&, const TransformOperations&);

} // namespace WebCore
