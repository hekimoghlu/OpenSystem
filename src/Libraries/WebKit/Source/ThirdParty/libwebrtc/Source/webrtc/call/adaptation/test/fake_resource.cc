/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#include "call/adaptation/test/fake_resource.h"

#include <algorithm>
#include <utility>

#include "absl/strings/string_view.h"
#include "api/make_ref_counted.h"

namespace webrtc {

// static
rtc::scoped_refptr<FakeResource> FakeResource::Create(absl::string_view name) {
  return rtc::make_ref_counted<FakeResource>(name);
}

FakeResource::FakeResource(absl::string_view name)
    : Resource(), name_(name), listener_(nullptr) {}

FakeResource::~FakeResource() {}

void FakeResource::SetUsageState(ResourceUsageState usage_state) {
  if (listener_) {
    listener_->OnResourceUsageStateMeasured(rtc::scoped_refptr<Resource>(this),
                                            usage_state);
  }
}

std::string FakeResource::Name() const {
  return name_;
}

void FakeResource::SetResourceListener(ResourceListener* listener) {
  listener_ = listener;
}

}  // namespace webrtc
