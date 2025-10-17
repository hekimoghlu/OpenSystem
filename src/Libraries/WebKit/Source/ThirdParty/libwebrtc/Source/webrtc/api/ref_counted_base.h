/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#ifndef API_REF_COUNTED_BASE_H_
#define API_REF_COUNTED_BASE_H_

#include <type_traits>

#include "api/ref_count.h"
#include "rtc_base/ref_counter.h"

namespace webrtc {

class RefCountedBase {
 public:
  RefCountedBase() = default;

  RefCountedBase(const RefCountedBase&) = delete;
  RefCountedBase& operator=(const RefCountedBase&) = delete;

  void AddRef() const { ref_count_.IncRef(); }
  RefCountReleaseStatus Release() const {
    const auto status = ref_count_.DecRef();
    if (status == RefCountReleaseStatus::kDroppedLastRef) {
      delete this;
    }
    return status;
  }

 protected:
  // Provided for internal webrtc subclasses for corner cases where it's
  // necessary to know whether or not a reference is exclusively held.
  bool HasOneRef() const { return ref_count_.HasOneRef(); }

  virtual ~RefCountedBase() = default;

 private:
  mutable webrtc::webrtc_impl::RefCounter ref_count_{0};
};

// Template based version of `RefCountedBase` for simple implementations that do
// not need (or want) destruction via virtual destructor or the overhead of a
// vtable.
//
// To use:
//   struct MyInt : public rtc::RefCountedNonVirtual<MyInt>  {
//     int foo_ = 0;
//   };
//
//   rtc::scoped_refptr<MyInt> my_int(new MyInt());
//
// sizeof(MyInt) on a 32 bit system would then be 8, int + refcount and no
// vtable generated.
template <typename T>
class RefCountedNonVirtual {
 public:
  RefCountedNonVirtual() = default;

  RefCountedNonVirtual(const RefCountedNonVirtual&) = delete;
  RefCountedNonVirtual& operator=(const RefCountedNonVirtual&) = delete;

  void AddRef() const { ref_count_.IncRef(); }
  RefCountReleaseStatus Release() const {
    // If you run into this assert, T has virtual methods. There are two
    // options:
    // 1) The class doesn't actually need virtual methods, the type is complete
    //    so the virtual attribute(s) can be removed.
    // 2) The virtual methods are a part of the design of the class. In this
    //    case you can consider using `RefCountedBase` instead or alternatively
    //    use `rtc::RefCountedObject`.
    static_assert(!std::is_polymorphic<T>::value,
                  "T has virtual methods. RefCountedBase is a better fit.");
    const auto status = ref_count_.DecRef();
    if (status == RefCountReleaseStatus::kDroppedLastRef) {
      delete static_cast<const T*>(this);
    }
    return status;
  }

 protected:
  // Provided for internal webrtc subclasses for corner cases where it's
  // necessary to know whether or not a reference is exclusively held.
  bool HasOneRef() const { return ref_count_.HasOneRef(); }

  ~RefCountedNonVirtual() = default;

 private:
  mutable webrtc::webrtc_impl::RefCounter ref_count_{0};
};

}  // namespace webrtc

// Backwards compatibe aliases.
// TODO: https://issues.webrtc.org/42225969 - deprecate and remove.
namespace rtc {
using RefCountedBase = webrtc::RefCountedBase;
template <typename T>
using RefCountedNonVirtual = webrtc::RefCountedNonVirtual<T>;
}  // namespace rtc

#endif  // API_REF_COUNTED_BASE_H_
