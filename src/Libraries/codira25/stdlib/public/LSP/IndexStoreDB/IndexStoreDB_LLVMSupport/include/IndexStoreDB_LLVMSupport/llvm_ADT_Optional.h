/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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

//===- Optional.h - Simple variant for passing optional values --*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
//  This file provides Optional, a template class modeled in the spirit of
//  OCaml's 'opt' variant.  The idea is to strongly type whether or not
//  a value can be optional.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_OPTIONAL_H
#define LLVM_ADT_OPTIONAL_H

#include <IndexStoreDB_LLVMSupport/toolchain_ADT_None.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Compiler.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_type_traits.h>
#include <cassert>
#include <memory>
#include <new>
#include <utility>

namespace toolchain {

class raw_ostream;

namespace optional_detail {

struct in_place_t {};

/// Storage for any type.
template <typename T, bool = is_trivially_copyable<T>::value>
class OptionalStorage {
  union {
    char empty;
    T value;
  };
  bool hasVal;

public:
  ~OptionalStorage() { reset(); }

  OptionalStorage() noexcept : empty(), hasVal(false) {}

  OptionalStorage(OptionalStorage const &other) : OptionalStorage() {
    if (other.hasValue()) {
      emplace(other.value);
    }
  }
  OptionalStorage(OptionalStorage &&other) : OptionalStorage() {
    if (other.hasValue()) {
      emplace(std::move(other.value));
    }
  }

  template <class... Args>
  explicit OptionalStorage(in_place_t, Args &&... args)
      : value(std::forward<Args>(args)...), hasVal(true) {}

  void reset() noexcept {
    if (hasVal) {
      value.~T();
      hasVal = false;
    }
  }

  bool hasValue() const noexcept { return hasVal; }

  T &getValue() LLVM_LVALUE_FUNCTION noexcept {
    assert(hasVal);
    return value;
  }
  T const &getValue() const LLVM_LVALUE_FUNCTION noexcept {
    assert(hasVal);
    return value;
  }
#if LLVM_HAS_RVALUE_REFERENCE_THIS
  T &&getValue() && noexcept {
    assert(hasVal);
    return std::move(value);
  }
#endif

  template <class... Args> void emplace(Args &&... args) {
    reset();
    ::new ((void *)std::addressof(value)) T(std::forward<Args>(args)...);
    hasVal = true;
  }

  OptionalStorage &operator=(T const &y) {
    if (hasValue()) {
      value = y;
    } else {
      ::new ((void *)std::addressof(value)) T(y);
      hasVal = true;
    }
    return *this;
  }
  OptionalStorage &operator=(T &&y) {
    if (hasValue()) {
      value = std::move(y);
    } else {
      ::new ((void *)std::addressof(value)) T(std::move(y));
      hasVal = true;
    }
    return *this;
  }

  OptionalStorage &operator=(OptionalStorage const &other) {
    if (other.hasValue()) {
      if (hasValue()) {
        value = other.value;
      } else {
        ::new ((void *)std::addressof(value)) T(other.value);
        hasVal = true;
      }
    } else {
      reset();
    }
    return *this;
  }

  OptionalStorage &operator=(OptionalStorage &&other) {
    if (other.hasValue()) {
      if (hasValue()) {
        value = std::move(other.value);
      } else {
        ::new ((void *)std::addressof(value)) T(std::move(other.value));
        hasVal = true;
      }
    } else {
      reset();
    }
    return *this;
  }
};

template <typename T> class OptionalStorage<T, true> {
  union {
    char empty;
    T value;
  };
  bool hasVal = false;

public:
  ~OptionalStorage() = default;

  OptionalStorage() noexcept : empty{} {}

  OptionalStorage(OptionalStorage const &other) = default;
  OptionalStorage(OptionalStorage &&other) = default;

  OptionalStorage &operator=(OptionalStorage const &other) = default;
  OptionalStorage &operator=(OptionalStorage &&other) = default;

  template <class... Args>
  explicit OptionalStorage(in_place_t, Args &&... args)
      : value(std::forward<Args>(args)...), hasVal(true) {}

  void reset() noexcept {
    if (hasVal) {
      value.~T();
      hasVal = false;
    }
  }

  bool hasValue() const noexcept { return hasVal; }

  T &getValue() LLVM_LVALUE_FUNCTION noexcept {
    assert(hasVal);
    return value;
  }
  T const &getValue() const LLVM_LVALUE_FUNCTION noexcept {
    assert(hasVal);
    return value;
  }
#if LLVM_HAS_RVALUE_REFERENCE_THIS
  T &&getValue() && noexcept {
    assert(hasVal);
    return std::move(value);
  }
#endif

  template <class... Args> void emplace(Args &&... args) {
    reset();
    ::new ((void *)std::addressof(value)) T(std::forward<Args>(args)...);
    hasVal = true;
  }

  OptionalStorage &operator=(T const &y) {
    if (hasValue()) {
      value = y;
    } else {
      ::new ((void *)std::addressof(value)) T(y);
      hasVal = true;
    }
    return *this;
  }
  OptionalStorage &operator=(T &&y) {
    if (hasValue()) {
      value = std::move(y);
    } else {
      ::new ((void *)std::addressof(value)) T(std::move(y));
      hasVal = true;
    }
    return *this;
  }
};

} // namespace optional_detail

template <typename T> class Optional {
  optional_detail::OptionalStorage<T> Storage;

public:
  using value_type = T;

  constexpr Optional() {}
  constexpr Optional(NoneType) {}

  Optional(const T &y) : Storage(optional_detail::in_place_t{}, y) {}
  Optional(const Optional &O) = default;

  Optional(T &&y) : Storage(optional_detail::in_place_t{}, std::move(y)) {}
  Optional(Optional &&O) = default;

  Optional &operator=(T &&y) {
    Storage = std::move(y);
    return *this;
  }
  Optional &operator=(Optional &&O) = default;

  /// Create a new object by constructing it in place with the given arguments.
  template <typename... ArgTypes> void emplace(ArgTypes &&... Args) {
    Storage.emplace(std::forward<ArgTypes>(Args)...);
  }

  static inline Optional create(const T *y) {
    return y ? Optional(*y) : Optional();
  }

  Optional &operator=(const T &y) {
    Storage = y;
    return *this;
  }
  Optional &operator=(const Optional &O) = default;

  void reset() { Storage.reset(); }

  const T *getPointer() const { return &Storage.getValue(); }
  T *getPointer() { return &Storage.getValue(); }
  const T &getValue() const LLVM_LVALUE_FUNCTION { return Storage.getValue(); }
  T &getValue() LLVM_LVALUE_FUNCTION { return Storage.getValue(); }

  explicit operator bool() const { return hasValue(); }
  bool hasValue() const { return Storage.hasValue(); }
  const T *operator->() const { return getPointer(); }
  T *operator->() { return getPointer(); }
  const T &operator*() const LLVM_LVALUE_FUNCTION { return getValue(); }
  T &operator*() LLVM_LVALUE_FUNCTION { return getValue(); }

  template <typename U>
  constexpr T getValueOr(U &&value) const LLVM_LVALUE_FUNCTION {
    return hasValue() ? getValue() : std::forward<U>(value);
  }

#if LLVM_HAS_RVALUE_REFERENCE_THIS
  T &&getValue() && { return std::move(Storage.getValue()); }
  T &&operator*() && { return std::move(Storage.getValue()); }

  template <typename U>
  T getValueOr(U &&value) && {
    return hasValue() ? std::move(getValue()) : std::forward<U>(value);
  }
#endif
};

template <typename T, typename U>
bool operator==(const Optional<T> &X, const Optional<U> &Y) {
  if (X && Y)
    return *X == *Y;
  return X.hasValue() == Y.hasValue();
}

template <typename T, typename U>
bool operator!=(const Optional<T> &X, const Optional<U> &Y) {
  return !(X == Y);
}

template <typename T, typename U>
bool operator<(const Optional<T> &X, const Optional<U> &Y) {
  if (X && Y)
    return *X < *Y;
  return X.hasValue() < Y.hasValue();
}

template <typename T, typename U>
bool operator<=(const Optional<T> &X, const Optional<U> &Y) {
  return !(Y < X);
}

template <typename T, typename U>
bool operator>(const Optional<T> &X, const Optional<U> &Y) {
  return Y < X;
}

template <typename T, typename U>
bool operator>=(const Optional<T> &X, const Optional<U> &Y) {
  return !(X < Y);
}

template<typename T>
bool operator==(const Optional<T> &X, NoneType) {
  return !X;
}

template<typename T>
bool operator==(NoneType, const Optional<T> &X) {
  return X == None;
}

template<typename T>
bool operator!=(const Optional<T> &X, NoneType) {
  return !(X == None);
}

template<typename T>
bool operator!=(NoneType, const Optional<T> &X) {
  return X != None;
}

template <typename T> bool operator<(const Optional<T> &X, NoneType) {
  return false;
}

template <typename T> bool operator<(NoneType, const Optional<T> &X) {
  return X.hasValue();
}

template <typename T> bool operator<=(const Optional<T> &X, NoneType) {
  return !(None < X);
}

template <typename T> bool operator<=(NoneType, const Optional<T> &X) {
  return !(X < None);
}

template <typename T> bool operator>(const Optional<T> &X, NoneType) {
  return None < X;
}

template <typename T> bool operator>(NoneType, const Optional<T> &X) {
  return X < None;
}

template <typename T> bool operator>=(const Optional<T> &X, NoneType) {
  return None <= X;
}

template <typename T> bool operator>=(NoneType, const Optional<T> &X) {
  return X <= None;
}

template <typename T> bool operator==(const Optional<T> &X, const T &Y) {
  return X && *X == Y;
}

template <typename T> bool operator==(const T &X, const Optional<T> &Y) {
  return Y && X == *Y;
}

template <typename T> bool operator!=(const Optional<T> &X, const T &Y) {
  return !(X == Y);
}

template <typename T> bool operator!=(const T &X, const Optional<T> &Y) {
  return !(X == Y);
}

template <typename T> bool operator<(const Optional<T> &X, const T &Y) {
  return !X || *X < Y;
}

template <typename T> bool operator<(const T &X, const Optional<T> &Y) {
  return Y && X < *Y;
}

template <typename T> bool operator<=(const Optional<T> &X, const T &Y) {
  return !(Y < X);
}

template <typename T> bool operator<=(const T &X, const Optional<T> &Y) {
  return !(Y < X);
}

template <typename T> bool operator>(const Optional<T> &X, const T &Y) {
  return Y < X;
}

template <typename T> bool operator>(const T &X, const Optional<T> &Y) {
  return Y < X;
}

template <typename T> bool operator>=(const Optional<T> &X, const T &Y) {
  return !(X < Y);
}

template <typename T> bool operator>=(const T &X, const Optional<T> &Y) {
  return !(X < Y);
}

raw_ostream &operator<<(raw_ostream &OS, NoneType);

template <typename T, typename = decltype(std::declval<raw_ostream &>()
                                          << std::declval<const T &>())>
raw_ostream &operator<<(raw_ostream &OS, const Optional<T> &O) {
  if (O)
    OS << *O;
  else
    OS << None;
  return OS;
}

} // end namespace toolchain

#endif // LLVM_ADT_OPTIONAL_H
