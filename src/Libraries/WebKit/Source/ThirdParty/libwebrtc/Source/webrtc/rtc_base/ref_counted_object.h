/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#ifndef RTC_BASE_REF_COUNTED_OBJECT_H_
#define RTC_BASE_REF_COUNTED_OBJECT_H_

#include "api/scoped_refptr.h"
#include "rtc_base/ref_count.h"
#include "rtc_base/ref_counter.h"

namespace webrtc {

template <class T>
class RefCountedObject : public T {
 public:
  RefCountedObject() {}

  RefCountedObject(const RefCountedObject&) = delete;
  RefCountedObject& operator=(const RefCountedObject&) = delete;

  template <class P0>
  explicit RefCountedObject(P0&& p0) : T(std::forward<P0>(p0)) {}

  template <class P0, class P1, class... Args>
  RefCountedObject(P0&& p0, P1&& p1, Args&&... args)
      : T(std::forward<P0>(p0),
          std::forward<P1>(p1),
          std::forward<Args>(args)...) {}

  void AddRef() const override { ref_count_.IncRef(); }

  RefCountReleaseStatus Release() const override {
    const auto status = ref_count_.DecRef();
    if (status == RefCountReleaseStatus::kDroppedLastRef) {
      delete this;
    }
    return status;
  }

  // Return whether the reference count is one. If the reference count is used
  // in the conventional way, a reference count of 1 implies that the current
  // thread owns the reference and no other thread shares it. This call
  // performs the test for a reference count of one, and performs the memory
  // barrier needed for the owning thread to act on the object, knowing that it
  // has exclusive access to the object.
  virtual bool HasOneRef() const { return ref_count_.HasOneRef(); }

 protected:
  ~RefCountedObject() override {}

  mutable webrtc::webrtc_impl::RefCounter ref_count_{0};
};

template <class T>
class FinalRefCountedObject final : public T {
 public:
  using T::T;
  // Above using declaration propagates a default move constructor
  // FinalRefCountedObject(FinalRefCountedObject&& other), but we also need
  // move construction from T.
  explicit FinalRefCountedObject(T&& other) : T(std::move(other)) {}
  FinalRefCountedObject(const FinalRefCountedObject&) = delete;
  FinalRefCountedObject& operator=(const FinalRefCountedObject&) = delete;

  void AddRef() const { ref_count_.IncRef(); }
  RefCountReleaseStatus Release() const {
    const auto status = ref_count_.DecRef();
    if (status == RefCountReleaseStatus::kDroppedLastRef) {
      delete this;
    }
    return status;
  }
  bool HasOneRef() const { return ref_count_.HasOneRef(); }

 private:
  ~FinalRefCountedObject() = default;

  mutable webrtc::webrtc_impl::RefCounter ref_count_{0};
};

}  // namespace webrtc

// Backwards compatibe aliases.
// TODO: https://issues.webrtc.org/42225969 - deprecate and remove.
namespace rtc {
using ::webrtc::FinalRefCountedObject;
using ::webrtc::RefCountedObject;
}  // namespace rtc

#endif  // RTC_BASE_REF_COUNTED_OBJECT_H_
