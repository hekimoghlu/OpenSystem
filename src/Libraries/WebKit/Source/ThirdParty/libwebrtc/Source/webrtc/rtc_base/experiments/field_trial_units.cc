/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#include "rtc_base/experiments/field_trial_units.h"

#include <stdio.h>

#include <limits>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"

// Large enough to fit "seconds", the longest supported unit name.
#define RTC_TRIAL_UNIT_LENGTH_STR "7"
#define RTC_TRIAL_UNIT_SIZE 8

namespace webrtc {
namespace {

struct ValueWithUnit {
  double value;
  std::string unit;
};

std::optional<ValueWithUnit> ParseValueWithUnit(absl::string_view str) {
  if (str == "inf") {
    return ValueWithUnit{std::numeric_limits<double>::infinity(), ""};
  } else if (str == "-inf") {
    return ValueWithUnit{-std::numeric_limits<double>::infinity(), ""};
  } else {
    double double_val;
    char unit_char[RTC_TRIAL_UNIT_SIZE];
    unit_char[0] = 0;
    if (sscanf(std::string(str).c_str(), "%lf%" RTC_TRIAL_UNIT_LENGTH_STR "s",
               &double_val, unit_char) >= 1) {
      return ValueWithUnit{double_val, unit_char};
    }
  }
  return std::nullopt;
}
}  // namespace

template <>
std::optional<DataRate> ParseTypedParameter<DataRate>(absl::string_view str) {
  std::optional<ValueWithUnit> result = ParseValueWithUnit(str);
  if (result) {
    if (result->unit.empty() || result->unit == "kbps") {
      return DataRate::KilobitsPerSec(result->value);
    } else if (result->unit == "bps") {
      return DataRate::BitsPerSec(result->value);
    }
  }
  return std::nullopt;
}

template <>
std::optional<DataSize> ParseTypedParameter<DataSize>(absl::string_view str) {
  std::optional<ValueWithUnit> result = ParseValueWithUnit(str);
  if (result) {
    if (result->unit.empty() || result->unit == "bytes")
      return DataSize::Bytes(result->value);
  }
  return std::nullopt;
}

template <>
std::optional<TimeDelta> ParseTypedParameter<TimeDelta>(absl::string_view str) {
  std::optional<ValueWithUnit> result = ParseValueWithUnit(str);
  if (result) {
    if (result->unit == "s" || result->unit == "seconds") {
      return TimeDelta::Seconds(result->value);
    } else if (result->unit == "us") {
      return TimeDelta::Micros(result->value);
    } else if (result->unit.empty() || result->unit == "ms") {
      return TimeDelta::Millis(result->value);
    }
  }
  return std::nullopt;
}

template <>
std::optional<std::optional<DataRate>>
ParseTypedParameter<std::optional<DataRate>>(absl::string_view str) {
  return ParseOptionalParameter<DataRate>(str);
}
template <>
std::optional<std::optional<DataSize>>
ParseTypedParameter<std::optional<DataSize>>(absl::string_view str) {
  return ParseOptionalParameter<DataSize>(str);
}
template <>
std::optional<std::optional<TimeDelta>>
ParseTypedParameter<std::optional<TimeDelta>>(absl::string_view str) {
  return ParseOptionalParameter<TimeDelta>(str);
}

template class FieldTrialParameter<DataRate>;
template class FieldTrialParameter<DataSize>;
template class FieldTrialParameter<TimeDelta>;

template class FieldTrialConstrained<DataRate>;
template class FieldTrialConstrained<DataSize>;
template class FieldTrialConstrained<TimeDelta>;

template class FieldTrialOptional<DataRate>;
template class FieldTrialOptional<DataSize>;
template class FieldTrialOptional<TimeDelta>;
}  // namespace webrtc
