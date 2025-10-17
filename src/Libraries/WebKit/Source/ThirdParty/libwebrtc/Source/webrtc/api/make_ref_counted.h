/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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
#ifndef API_MAKE_REF_COUNTED_H_
#define API_MAKE_REF_COUNTED_H_

#include <type_traits>
#include <utility>

#include "absl/base/nullability.h"
#include "api/ref_count.h"
#include "api/scoped_refptr.h"
#include "rtc_base/ref_counted_object.h"

namespace webrtc {

namespace webrtc_make_ref_counted_internal {
// Determines if the given class has AddRef and Release methods.
template <typename T>
class HasAddRefAndRelease {
 private:
  template <typename C,
            decltype(std::declval<C>().AddRef())* = nullptr,
            decltype(std::declval<C>().Release())* = nullptr>
  static int Test(int);
  template <typename>
  static char Test(...);

 public:
  static constexpr bool value = std::is_same_v<decltype(Test<T>(0)), int>;
};
}  // namespace webrtc_make_ref_counted_internal

// General utilities for constructing a reference counted class and the
// appropriate reference count implementation for that class.
//
// These utilities select either the `RefCountedObject` implementation or
// `FinalRefCountedObject` depending on whether the to-be-shared class is
// derived from the RefCountInterface interface or not (respectively).

// `make_ref_counted`:
//
// Use this when you want to construct a reference counted object of type T and
// get a `scoped_refptr<>` back. Example:
//
//   auto p = make_ref_counted<Foo>("bar", 123);
//
// For a class that inherits from RefCountInterface, this is equivalent to:
//
//   auto p = scoped_refptr<Foo>(new RefCountedObject<Foo>("bar", 123));
//
// If the class does not inherit from RefCountInterface, but does have
// AddRef/Release methods (so a T* is convertible to rtc::scoped_refptr), this
// is equivalent to just
//
//   auto p = scoped_refptr<Foo>(new Foo("bar", 123));
//
// Otherwise, the example is equivalent to:
//
//   auto p = scoped_refptr<FinalRefCountedObject<Foo>>(
//       new FinalRefCountedObject<Foo>("bar", 123));
//
// In these cases, `make_ref_counted` reduces the amount of boilerplate code but
// also helps with the most commonly intended usage of RefCountedObject whereby
// methods for reference counting, are virtual and designed to satisfy the need
// of an interface. When such a need does not exist, it is more efficient to use
// the `FinalRefCountedObject` template, which does not add the vtable overhead.
//
// Note that in some cases, using RefCountedObject directly may still be what's
// needed.

// `make_ref_counted` for abstract classes that are convertible to
// RefCountInterface. The is_abstract requirement rejects classes that inherit
// both RefCountInterface and RefCounted object, which is a a discouraged
// pattern, and would result in double inheritance of RefCountedObject if this
// template was applied.
template <
    typename T,
    typename... Args,
    typename std::enable_if<std::is_convertible_v<T*, RefCountInterface*> &&
                                std::is_abstract_v<T>,
                            T>::type* = nullptr>
absl::Nonnull<scoped_refptr<T>> make_ref_counted(Args&&... args) {
  return scoped_refptr<T>(new RefCountedObject<T>(std::forward<Args>(args)...));
}

// `make_ref_counted` for complete classes that are not convertible to
// RefCountInterface and already carry a ref count.
template <
    typename T,
    typename... Args,
    typename std::enable_if<
        !std::is_convertible_v<T*, RefCountInterface*> &&
            webrtc_make_ref_counted_internal::HasAddRefAndRelease<T>::value,
        T>::type* = nullptr>
absl::Nonnull<scoped_refptr<T>> make_ref_counted(Args&&... args) {
  return scoped_refptr<T>(new T(std::forward<Args>(args)...));
}

// `make_ref_counted` for complete classes that are not convertible to
// RefCountInterface and have no ref count of their own.
template <
    typename T,
    typename... Args,
    typename std::enable_if<
        !std::is_convertible_v<T*, RefCountInterface*> &&
            !webrtc_make_ref_counted_internal::HasAddRefAndRelease<T>::value,

        T>::type* = nullptr>
absl::Nonnull<scoped_refptr<FinalRefCountedObject<T>>> make_ref_counted(
    Args&&... args) {
  return scoped_refptr<FinalRefCountedObject<T>>(
      new FinalRefCountedObject<T>(std::forward<Args>(args)...));
}

}  // namespace webrtc

namespace rtc {
// Backwards compatibe alias.
// TODO: bugs.webrtc.org/42225969 - deprecate and remove.
using ::webrtc::make_ref_counted;
}  // namespace rtc

#endif  // API_MAKE_REF_COUNTED_H_
