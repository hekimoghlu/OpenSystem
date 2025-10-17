/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#ifndef CALL_ADAPTATION_TEST_FAKE_ADAPTATION_CONSTRAINT_H_
#define CALL_ADAPTATION_TEST_FAKE_ADAPTATION_CONSTRAINT_H_

#include <string>

#include "absl/strings/string_view.h"
#include "call/adaptation/adaptation_constraint.h"

namespace webrtc {

class FakeAdaptationConstraint : public AdaptationConstraint {
 public:
  explicit FakeAdaptationConstraint(absl::string_view name);
  ~FakeAdaptationConstraint() override;

  void set_is_adaptation_up_allowed(bool is_adaptation_up_allowed);

  // AdaptationConstraint implementation.
  std::string Name() const override;
  bool IsAdaptationUpAllowed(
      const VideoStreamInputState& input_state,
      const VideoSourceRestrictions& restrictions_before,
      const VideoSourceRestrictions& restrictions_after) const override;

 private:
  const std::string name_;
  bool is_adaptation_up_allowed_;
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_TEST_FAKE_ADAPTATION_CONSTRAINT_H_
