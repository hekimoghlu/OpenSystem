/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#ifndef CALL_ADAPTATION_TEST_FAKE_RESOURCE_H_
#define CALL_ADAPTATION_TEST_FAKE_RESOURCE_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/adaptation/resource.h"
#include "api/scoped_refptr.h"

namespace webrtc {

// Fake resource used for testing.
class FakeResource : public Resource {
 public:
  static rtc::scoped_refptr<FakeResource> Create(absl::string_view name);

  explicit FakeResource(absl::string_view name);
  ~FakeResource() override;

  void SetUsageState(ResourceUsageState usage_state);

  // Resource implementation.
  std::string Name() const override;
  void SetResourceListener(ResourceListener* listener) override;

 private:
  const std::string name_;
  ResourceListener* listener_;
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_TEST_FAKE_RESOURCE_H_
