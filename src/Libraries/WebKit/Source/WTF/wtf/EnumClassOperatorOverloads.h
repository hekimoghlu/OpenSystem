/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#define OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, op, enableExpression) \
    template<typename T> \
    constexpr auto operator op(enumName enumEntry, T value) -> std::enable_if_t<(enableExpression), T> \
    { \
        return static_cast<T>(enumEntry) op value; \
    } \
    \
    template<typename T> \
    constexpr auto operator op(T value, enumName enumEntry) -> std::enable_if_t<(enableExpression), T> \
    { \
        return value op static_cast<T>(enumEntry); \
    } \

#define OVERLOAD_MATH_OPERATORS_FOR_ENUM_CLASS_WHEN(enumName, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, +, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, -, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, *, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, /, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, %, enableExpression) \

#define OVERLOAD_MATH_OPERATORS_FOR_ENUM_CLASS_WITH_INTEGRALS(enumName) OVERLOAD_MATH_OPERATORS_FOR_ENUM_CLASS_WHEN(enumName, std::is_integral_v<T>)

#define OVERLOAD_RELATIONAL_OPERATORS_FOR_ENUM_CLASS_WHEN(enumName, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, ==, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, !=, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, <, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, <=, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, >, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, >=, enableExpression) \

#define OVERLOAD_RELATIONAL_OPERATORS_FOR_ENUM_CLASS_WITH_INTEGRALS(enumName) OVERLOAD_RELATIONAL_OPERATORS_FOR_ENUM_CLASS_WHEN(enumName, std::is_integral_v<T>)

#define OVERLOAD_BITWISE_OPERATORS_FOR_ENUM_CLASS_WHEN(enumName, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, |, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, &, enableExpression) \
    OVERLOAD_OPERATOR_FOR_ENUM_CLASS_WHEN(enumName, ^, enableExpression) \

#define OVERLOAD_BITWISE_OPERATORS_FOR_ENUM_CLASS_WITH_INTERGRALS(enumName) OVERLOAD_BITWISE_OPERATORS_FOR_ENUM_CLASS_WHEN(enumName, std::is_integral_v<T>)
