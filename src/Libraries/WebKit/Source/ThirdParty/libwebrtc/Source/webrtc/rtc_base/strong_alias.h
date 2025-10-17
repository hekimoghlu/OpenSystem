/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#ifndef RTC_BASE_STRONG_ALIAS_H_
#define RTC_BASE_STRONG_ALIAS_H_

#include <type_traits>
#include <utility>

namespace webrtc {

// This is a copy of
// https://source.chromium.org/chromium/chromium/src/+/main:base/types/strong_alias.h
// as the API (and internals) are using type-safe integral identifiers, but this
// library can't depend on that file. The ostream operator has been removed
// per WebRTC library conventions, and the underlying type is exposed.

template <typename TagType, typename TheUnderlyingType>
class StrongAlias {
 public:
  using UnderlyingType = TheUnderlyingType;
  constexpr StrongAlias() = default;
  constexpr explicit StrongAlias(const UnderlyingType& v) : value_(v) {}
  constexpr explicit StrongAlias(UnderlyingType&& v) noexcept
      : value_(std::move(v)) {}

  constexpr UnderlyingType* operator->() { return &value_; }
  constexpr const UnderlyingType* operator->() const { return &value_; }

  constexpr UnderlyingType& operator*() & { return value_; }
  constexpr const UnderlyingType& operator*() const& { return value_; }
  constexpr UnderlyingType&& operator*() && { return std::move(value_); }
  constexpr const UnderlyingType&& operator*() const&& {
    return std::move(value_);
  }

  constexpr UnderlyingType& value() & { return value_; }
  constexpr const UnderlyingType& value() const& { return value_; }
  constexpr UnderlyingType&& value() && { return std::move(value_); }
  constexpr const UnderlyingType&& value() const&& { return std::move(value_); }

  constexpr explicit operator const UnderlyingType&() const& { return value_; }

  constexpr bool operator==(const StrongAlias& other) const {
    return value_ == other.value_;
  }
  constexpr bool operator!=(const StrongAlias& other) const {
    return value_ != other.value_;
  }
  constexpr bool operator<(const StrongAlias& other) const {
    return value_ < other.value_;
  }
  constexpr bool operator<=(const StrongAlias& other) const {
    return value_ <= other.value_;
  }
  constexpr bool operator>(const StrongAlias& other) const {
    return value_ > other.value_;
  }
  constexpr bool operator>=(const StrongAlias& other) const {
    return value_ >= other.value_;
  }

 protected:
  UnderlyingType value_;
};

}  // namespace webrtc

#endif  // RTC_BASE_STRONG_ALIAS_H_
