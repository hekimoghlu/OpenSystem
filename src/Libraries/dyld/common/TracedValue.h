/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#ifndef __TRACED_VALUE__
#define __TRACED_VALUE__

// std
#include <memory>

namespace ld {

/// TracedValue wrapps  options with some default value that might be explicity
/// overwritten. The wrapper is constructed with `isDefault` flag set to true,
/// but whenever a new value is assigned to an existing wrapper instance
/// isDefault becomes false too.
template<typename Val>
struct TracedValue {

  protected:
  Val val;

  public:
  bool isDefault;

  TracedValue() = delete;
  constexpr TracedValue(Val val, bool isDefault = true) noexcept: val(std::move(val)), isDefault(isDefault) {}
  constexpr TracedValue(TracedValue&&) = default;
  constexpr TracedValue(const TracedValue&) = default;

  constexpr operator Val() const { return val; }
  constexpr Val get() const { return val; }

  constexpr TracedValue& operator=(TracedValue&& other)
  {
    val = std::move(other.val);
    isDefault = false;
    return *this;
  }

  constexpr TracedValue& operator=(const TracedValue& other)
  {
    val = other.val;
    isDefault = false;
    return *this;
  }

  constexpr TracedValue& operator=(Val newValue)
  {
    val = std::move(newValue);
    isDefault = false;
    return *this;
  }

  // Set new value without changing the isDefault flag.
  constexpr void overwrite(Val newDefault)
  {
    val = std::move(newDefault);
  }

  constexpr void overwrite(Val newVal, bool newIsDefault) {
    val = std::move(newVal);
    isDefault = std::move(newIsDefault);
  }
};

struct TracedBool: public TracedValue<bool> {

  TracedBool() = delete;
  constexpr TracedBool(bool val, bool isDefault = true): TracedValue(val, isDefault) {}

  constexpr operator bool() const { return val; }
  constexpr bool isForceOn() const { return !isDefault && val; }
  constexpr bool isForceOff() const { return !isDefault && !val; }
};
}
#endif // __TRACED_VALUE__
