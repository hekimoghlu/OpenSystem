/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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
#include "rtc_base/weak_ptr.h"

// The implementation is borrowed from chromium except that it does not
// implement SupportsWeakPtr.

namespace rtc {
namespace internal {

void WeakReference::Flag::Invalidate() {
  RTC_DCHECK_RUN_ON(&checker_);
  is_valid_ = false;
}

bool WeakReference::Flag::IsValid() const {
  RTC_DCHECK_RUN_ON(&checker_);
  return is_valid_;
}

WeakReference::WeakReference() {}

WeakReference::WeakReference(const RefCountedFlag* flag) : flag_(flag) {}

WeakReference::~WeakReference() {}

WeakReference::WeakReference(WeakReference&& other) = default;

WeakReference::WeakReference(const WeakReference& other) = default;

bool WeakReference::is_valid() const {
  return flag_.get() && flag_->IsValid();
}

WeakReferenceOwner::WeakReferenceOwner() {}

WeakReferenceOwner::~WeakReferenceOwner() {
  Invalidate();
}

WeakReference WeakReferenceOwner::GetRef() const {
  // If we hold the last reference to the Flag then create a new one.
  if (!HasRefs())
    flag_ = new WeakReference::RefCountedFlag();

  return WeakReference(flag_.get());
}

void WeakReferenceOwner::Invalidate() {
  if (flag_.get()) {
    flag_->Invalidate();
    flag_ = nullptr;
  }
}

WeakPtrBase::WeakPtrBase() {}

WeakPtrBase::~WeakPtrBase() {}

WeakPtrBase::WeakPtrBase(const WeakReference& ref) : ref_(ref) {}

}  // namespace internal
}  // namespace rtc
