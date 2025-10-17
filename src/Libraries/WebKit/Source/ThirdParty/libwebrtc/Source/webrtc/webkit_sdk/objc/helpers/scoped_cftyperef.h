/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#ifndef SDK_OBJC_HELPERS_SCOPED_CFTYPEREF_H_
#define SDK_OBJC_HELPERS_SCOPED_CFTYPEREF_H_

#include <CoreFoundation/CoreFoundation.h>
namespace rtc {

// RETAIN: ScopedTypeRef should retain the object when it takes
// ownership.
// ASSUME: Assume the object already has already been retained.
// ScopedTypeRef takes over ownership.
enum class RetainPolicy { RETAIN, ASSUME };

namespace internal {
template <typename T>
struct CFTypeRefTraits {
  static T InvalidValue() { return nullptr; }
  static void Release(T ref) { CFRelease(ref); }
  static T Retain(T ref) {
    CFRetain(ref);
    return ref;
  }
};

template <typename T, typename Traits>
class ScopedTypeRef {
 public:
  ScopedTypeRef() : ptr_(Traits::InvalidValue()) {}
  explicit ScopedTypeRef(T ptr) : ptr_(ptr) {}
  ScopedTypeRef(T ptr, RetainPolicy policy) : ScopedTypeRef(ptr) {
    if (ptr_ && policy == RetainPolicy::RETAIN)
      Traits::Retain(ptr_);
  }

  ScopedTypeRef(const ScopedTypeRef<T, Traits>& rhs) : ptr_(rhs.ptr_) {
    if (ptr_)
      ptr_ = Traits::Retain(ptr_);
  }

  ~ScopedTypeRef() {
    if (ptr_) {
      Traits::Release(ptr_);
    }
  }

  T get() const { return ptr_; }
  T operator->() const { return ptr_; }
  explicit operator bool() const { return ptr_; }

  bool operator!() const { return !ptr_; }

  ScopedTypeRef& operator=(const T& rhs) {
    if (ptr_)
      Traits::Release(ptr_);
    ptr_ = rhs;
    return *this;
  }

  ScopedTypeRef& operator=(const ScopedTypeRef<T, Traits>& rhs) {
    reset(rhs.get(), RetainPolicy::RETAIN);
    return *this;
  }

  // This is intended to take ownership of objects that are
  // created by pass-by-pointer initializers.
  T* InitializeInto() {
    RTC_DCHECK(!ptr_);
    return &ptr_;
  }

  void reset(T ptr, RetainPolicy policy = RetainPolicy::ASSUME) {
    if (ptr && policy == RetainPolicy::RETAIN)
      Traits::Retain(ptr);
    if (ptr_)
      Traits::Release(ptr_);
    ptr_ = ptr;
  }

  T release() {
    T temp = ptr_;
    ptr_ = Traits::InvalidValue();
    return temp;
  }

 private:
  T ptr_;
};
}  // namespace internal

template <typename T>
using ScopedCFTypeRef =
    internal::ScopedTypeRef<T, internal::CFTypeRefTraits<T>>;

template <typename T>
static ScopedCFTypeRef<T> AdoptCF(T cftype) {
  return ScopedCFTypeRef<T>(cftype, RetainPolicy::RETAIN);
}

template <typename T>
static ScopedCFTypeRef<T> ScopedCF(T cftype) {
  return ScopedCFTypeRef<T>(cftype);
}

}  // namespace rtc

#endif  // SDK_OBJC_HELPERS_SCOPED_CFTYPEREF_H_
