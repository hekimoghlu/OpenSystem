/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

#include <type_traits>
#include <wtf/FixedVector.h>
#include <wtf/HashMap.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WGSL {

#if HAVE(FP16_HALF_SUPPORT)
using half = __fp16;
#else
// Wrap a struct around the supported fp16 type.
struct half {
#if PLATFORM(COCOA)
    using f16 = __fp16;
#else
    // _Float16 is the 16bit float type in C++23, and is often available
    // in compilers prior to C++23.
    using f16 = _Float16;
#endif
    half()
    {
    }

    // Constructor from an arithmetic type. Use a template here because the
    // explicit list of types may differ among platforms.
    template <typename A,
        std::enable_if_t<std::is_arithmetic_v<std::decay_t<A>>, bool> = true>
    half(const A& val)
        : value(static_cast<f16>(val)) { }

    // Constructor from a ConstantResult.
    template <typename C,
        std::enable_if_t<std::is_class_v<std::decay_t<C>>, bool> = true>
    half(const C& val)
        : value(val.value().getHalf().value) { }

    operator float() const
    {
        return static_cast<float>(value);
    }

    f16 value { 0 };
};
#endif

// A constant value might be:
// - a scalar
// - a vector
// - a matrix
// - a fixed-size array type
// - a structure
struct ConstantValue;

struct ConstantArray {
    ConstantArray(size_t size)
        : elements(size)
    {
    }

    ConstantArray(FixedVector<ConstantValue>&& elements)
        : elements(WTFMove(elements))
    {
    }

    size_t upperBound() { return elements.size(); }
    ConstantValue operator[](unsigned);

    FixedVector<ConstantValue> elements;
};

struct ConstantVector {
    ConstantVector(size_t size)
        : elements(size)
    {
    }

    ConstantVector(FixedVector<ConstantValue>&& elements)
        : elements(WTFMove(elements))
    {
    }

    size_t upperBound() { return elements.size(); }
    ConstantValue operator[](unsigned);

    FixedVector<ConstantValue> elements;
};

struct ConstantMatrix {
    ConstantMatrix(uint32_t columns, uint32_t rows)
        : columns(columns)
        , rows(rows)
        , elements(columns * rows)
    {
    }

    ConstantMatrix(uint32_t columns, uint32_t rows, const FixedVector<ConstantValue>& elements)
        : columns(columns)
        , rows(rows)
        , elements(elements)
    {
        RELEASE_ASSERT(elements.size() == columns * rows);
    }

    size_t upperBound() { return columns; }
    ConstantVector operator[](unsigned);

    uint32_t columns;
    uint32_t rows;
    FixedVector<ConstantValue> elements;
};

struct ConstantStruct {
    HashMap<String, ConstantValue> fields;
};

using BaseValue = std::variant<float, half, double, int32_t, uint32_t, int64_t, bool, ConstantArray, ConstantVector, ConstantMatrix, ConstantStruct>;
struct ConstantValue : BaseValue {
    ConstantValue() = default;

    using BaseValue::BaseValue;

    void dump(PrintStream&) const;

    bool isBool() const { return std::holds_alternative<bool>(*this); }
    bool isVector() const { return std::holds_alternative<ConstantVector>(*this); }
    bool isMatrix() const { return std::holds_alternative<ConstantMatrix>(*this); }
    bool isArray() const { return std::holds_alternative<ConstantArray>(*this); }

    bool toBool() const { return std::get<bool>(*this); }

    int64_t integerValue() const
    {
        if (auto* i32 = std::get_if<int32_t>(this))
            return *i32;
        if (auto* u32 = std::get_if<uint32_t>(this))
            return *u32;
        if (auto* abstractInt = std::get_if<int64_t>(this))
            return *abstractInt;
        RELEASE_ASSERT_NOT_REACHED();
    }
    half getHalf() const
    {
        if (auto* f32 = std::get_if<float>(this))
            return *f32;
        if (auto* f64 = std::get_if<double>(this))
            return *f64;
        RELEASE_ASSERT_NOT_REACHED();
    }

    const ConstantVector& toVector() const
    {
        return std::get<ConstantVector>(*this);
    }
};

template<typename To, typename From>
std::optional<To> convertInteger(From value)
{
    auto result = Checked<To, RecordOverflow>(value);
    if (UNLIKELY(result.hasOverflowed()))
        return std::nullopt;
    return { result.value() };
}

template<typename To, typename From>
std::optional<To> convertFloat(From value)
{
    static_assert(std::is_floating_point<To>::value || std::is_same<To, half>::value, "Result type is expected to be a floating point type: double, float, or half");

    static To max;
    static To lowest;
    if constexpr (std::is_floating_point<To>::value) {
        max = std::numeric_limits<To>::max();
        lowest = std::numeric_limits<To>::lowest();
    } else {
        max = 0x1.ffcp15;
        lowest = -max;
    }

    if (value > max)
        return std::nullopt;
    if (value < lowest)
        return std::nullopt;
    if (std::isnan(value))
        return std::nullopt;

    return { value };
}

} // namespace WGSL
