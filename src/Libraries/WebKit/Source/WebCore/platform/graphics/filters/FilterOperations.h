/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#include "CompositeOperation.h"
#include "FilterOperation.h"
#include <algorithm>
#include <wtf/ArgumentCoder.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

struct BlendingContext;

class FilterOperations {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(FilterOperations, WEBCORE_EXPORT);
public:
    using const_iterator = Vector<Ref<FilterOperation>>::const_iterator;
    using const_reverse_iterator = Vector<Ref<FilterOperation>>::const_reverse_iterator;
    using value_type = Vector<Ref<FilterOperation>>::value_type;

    FilterOperations() = default;
    WEBCORE_EXPORT explicit FilterOperations(Vector<Ref<FilterOperation>>&&);

    WEBCORE_EXPORT bool operator==(const FilterOperations&) const;

    FilterOperations clone() const
    {
        return FilterOperations { m_operations.map([](const auto& op) { return op->clone(); }) };
    }

    const_iterator begin() const { return m_operations.begin(); }
    const_iterator end() const { return m_operations.end(); }
    const_reverse_iterator rbegin() const { return m_operations.rbegin(); }
    const_reverse_iterator rend() const { return m_operations.rend(); }

    bool isEmpty() const { return m_operations.isEmpty(); }
    size_t size() const { return m_operations.size(); }
    const FilterOperation* at(size_t index) const { return index < m_operations.size() ? m_operations[index].ptr() : nullptr; }

    const Ref<FilterOperation>& operator[](size_t i) const { return m_operations[i]; }
    const Ref<FilterOperation>& first() const { return m_operations.first(); }
    const Ref<FilterOperation>& last() const { return m_operations.last(); }

    bool operationsMatch(const FilterOperations&) const;

    bool hasOutsets() const { return !outsets().isZero(); }
    IntOutsets outsets() const;

    bool hasFilterThatAffectsOpacity() const;
    bool hasFilterThatMovesPixels() const;
    bool hasFilterThatShouldBeRestrictedBySecurityOrigin() const;

    template<FilterOperation::Type Type>
    bool hasFilterOfType() const;

    bool hasReferenceFilter() const;
    bool isReferenceFilter() const;

    bool transformColor(Color&) const;
    bool inverseTransformColor(Color&) const;

    WEBCORE_EXPORT bool canInterpolate(const FilterOperations&, CompositeOperation) const;
    WEBCORE_EXPORT FilterOperations blend(const FilterOperations&, const BlendingContext&) const;

private:
    friend struct IPC::ArgumentCoder<FilterOperations, void>;
    WEBCORE_EXPORT friend WTF::TextStream& operator<<(WTF::TextStream&, const FilterOperations&);

    Vector<Ref<FilterOperation>> m_operations;
};

template<FilterOperation::Type type> bool FilterOperations::hasFilterOfType() const
{
    return std::ranges::any_of(m_operations, [](auto& op) { return op->type() == type; });
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const FilterOperations&);

} // namespace WebCore
