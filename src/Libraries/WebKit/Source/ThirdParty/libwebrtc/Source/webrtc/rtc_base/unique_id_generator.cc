/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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
#include "rtc_base/unique_id_generator.h"

#include <limits>
#include <vector>

#include "absl/strings/string_view.h"
#include "rtc_base/crypto_random.h"
#include "rtc_base/string_encode.h"
#include "rtc_base/string_to_number.h"

namespace rtc {

UniqueRandomIdGenerator::UniqueRandomIdGenerator() : known_ids_() {}
UniqueRandomIdGenerator::UniqueRandomIdGenerator(ArrayView<uint32_t> known_ids)
    : known_ids_(known_ids.begin(), known_ids.end()) {}

UniqueRandomIdGenerator::~UniqueRandomIdGenerator() = default;

uint32_t UniqueRandomIdGenerator::GenerateId() {
  webrtc::MutexLock lock(&mutex_);

  RTC_CHECK_LT(known_ids_.size(), std::numeric_limits<uint32_t>::max() - 1);
  while (true) {
    auto pair = known_ids_.insert(CreateRandomNonZeroId());
    if (pair.second) {
      return *pair.first;
    }
  }
}

bool UniqueRandomIdGenerator::AddKnownId(uint32_t value) {
  webrtc::MutexLock lock(&mutex_);
  return known_ids_.insert(value).second;
}

UniqueStringGenerator::UniqueStringGenerator() : unique_number_generator_() {}
UniqueStringGenerator::UniqueStringGenerator(ArrayView<std::string> known_ids) {
  for (const std::string& str : known_ids) {
    AddKnownId(str);
  }
}

UniqueStringGenerator::~UniqueStringGenerator() = default;

std::string UniqueStringGenerator::GenerateString() {
  return ToString(unique_number_generator_.GenerateNumber());
}

bool UniqueStringGenerator::AddKnownId(absl::string_view value) {
  // TODO(webrtc:13579): remove string copy here once absl::string_view version
  // of StringToNumber is available.
  std::optional<uint32_t> int_value =
      StringToNumber<uint32_t>(std::string(value));
  // The underlying generator works for uint32_t values, so if the provided
  // value is not a uint32_t it will never be generated anyway.
  if (int_value.has_value()) {
    return unique_number_generator_.AddKnownId(int_value.value());
  }
  return false;
}

}  // namespace rtc
