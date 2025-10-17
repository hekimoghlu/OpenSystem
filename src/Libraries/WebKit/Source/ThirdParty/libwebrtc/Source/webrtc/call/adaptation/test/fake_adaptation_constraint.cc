/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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
#include "call/adaptation/test/fake_adaptation_constraint.h"

#include <utility>

#include "absl/strings/string_view.h"

namespace webrtc {

FakeAdaptationConstraint::FakeAdaptationConstraint(absl::string_view name)
    : name_(name), is_adaptation_up_allowed_(true) {}

FakeAdaptationConstraint::~FakeAdaptationConstraint() = default;

void FakeAdaptationConstraint::set_is_adaptation_up_allowed(
    bool is_adaptation_up_allowed) {
  is_adaptation_up_allowed_ = is_adaptation_up_allowed;
}

std::string FakeAdaptationConstraint::Name() const {
  return name_;
}

bool FakeAdaptationConstraint::IsAdaptationUpAllowed(
    const VideoStreamInputState& /* input_state */,
    const VideoSourceRestrictions& /* restrictions_before */,
    const VideoSourceRestrictions& /* restrictions_after */) const {
  return is_adaptation_up_allowed_;
}

}  // namespace webrtc
