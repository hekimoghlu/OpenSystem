/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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

#ifndef TEST_INTEROP_CXX_STDLIB_INPUTS_STD_OPTIONAL_H
#define TEST_INTEROP_CXX_STDLIB_INPUTS_STD_OPTIONAL_H

#include <optional>
#include <string>

using StdOptionalInt = std::optional<int>;
using StdOptionalBool = std::optional<bool>;
using StdOptionalString = std::optional<std::string>;
using StdOptionalOptionalInt = std::optional<std::optional<int>>;

struct HasConstexprCtor {
  int value;
  constexpr HasConstexprCtor(int value) : value(value) {}
  constexpr HasConstexprCtor(const HasConstexprCtor &other) = default;
  constexpr HasConstexprCtor(HasConstexprCtor &&other) = default;
};
using StdOptionalHasConstexprCtor = std::optional<HasConstexprCtor>;

struct HasDeletedMoveCtor {
  int value;
  HasDeletedMoveCtor(int value) : value(value) {}
  HasDeletedMoveCtor(const HasDeletedMoveCtor &other) : value(other.value) {}
  HasDeletedMoveCtor(HasDeletedMoveCtor &&other) = delete;
};
using StdOptionalHasDeletedMoveCtor = std::optional<HasDeletedMoveCtor>;

inline StdOptionalInt getNonNilOptional() { return {123}; }

inline StdOptionalInt getNilOptional() { return {std::nullopt}; }

inline bool takesOptionalInt(std::optional<int> arg) { return (bool)arg; }
inline bool takesOptionalString(std::optional<std::string> arg) { return (bool)arg; }

#endif // TEST_INTEROP_CXX_STDLIB_INPUTS_STD_OPTIONAL_H
