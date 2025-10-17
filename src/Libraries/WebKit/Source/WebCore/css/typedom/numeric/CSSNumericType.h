/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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

#include "CSSNumericBaseType.h"
#include <optional>
#include <wtf/Markable.h>

namespace WebCore {

class CSSNumericValue;
enum class CSSUnitType : uint8_t;

// https://drafts.css-houdini.org/css-typed-om/#dom-cssnumericvalue-type
class CSSNumericType {
public:
    using BaseTypeStorage = Markable<int, IntegralMarkableTraits<int, std::numeric_limits<int>::min()>>;
    BaseTypeStorage length;
    BaseTypeStorage angle;
    BaseTypeStorage time;
    BaseTypeStorage frequency;
    BaseTypeStorage resolution;
    BaseTypeStorage flex;
    BaseTypeStorage percent;
    Markable<CSSNumericBaseType, EnumMarkableTraits<CSSNumericBaseType>> percentHint;

    static std::optional<CSSNumericType> create(CSSUnitType, int exponent = 1);
    friend bool operator==(const CSSNumericType&, const CSSNumericType&) = default;
    static std::optional<CSSNumericType> addTypes(const Vector<Ref<CSSNumericValue>>&);
    static std::optional<CSSNumericType> addTypes(CSSNumericType, CSSNumericType);
    static std::optional<CSSNumericType> multiplyTypes(const Vector<Ref<CSSNumericValue>>&);
    static std::optional<CSSNumericType> multiplyTypes(const CSSNumericType&, const CSSNumericType&);
    BaseTypeStorage& valueForType(CSSNumericBaseType);
    const BaseTypeStorage& valueForType(CSSNumericBaseType type) const { return const_cast<CSSNumericType*>(this)->valueForType(type); }
    void applyPercentHint(CSSNumericBaseType);
    size_t nonZeroEntryCount() const;

    template<CSSNumericBaseType type>
    bool matches() const
    {
        // https://drafts.css-houdini.org/css-typed-om/#cssnumericvalue-match
        return (type == CSSNumericBaseType::Percent || !percentHint)
            && nonZeroEntryCount() == 1
            && valueForType(type)
            && *valueForType(type);
    }

    bool matchesNumber() const
    {
        // https://drafts.css-houdini.org/css-typed-om/#cssnumericvalue-match
        return !nonZeroEntryCount() && !percentHint;
    }

    template<CSSNumericBaseType type>
    bool matchesTypeOrPercentage() const
    {
        return matches<type>() || matches<CSSNumericBaseType::Percent>();
    }

    String debugString() const;
};

} // namespace WebCore
