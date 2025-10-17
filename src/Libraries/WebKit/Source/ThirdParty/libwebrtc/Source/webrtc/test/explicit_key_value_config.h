/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
#ifndef TEST_EXPLICIT_KEY_VALUE_CONFIG_H_
#define TEST_EXPLICIT_KEY_VALUE_CONFIG_H_

#include <functional>
#include <map>
#include <string>

#include "absl/strings/string_view.h"
#include "api/field_trials_registry.h"

namespace webrtc {
namespace test {

class ExplicitKeyValueConfig : public FieldTrialsRegistry {
 public:
  explicit ExplicitKeyValueConfig(absl::string_view s);

 private:
  std::string GetValue(absl::string_view key) const override;

  // Unlike std::less<std::string>, std::less<> is transparent and allows
  // heterogeneous lookup directly with absl::string_view.
  std::map<std::string, std::string, std::less<>> key_value_map_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_EXPLICIT_KEY_VALUE_CONFIG_H_
