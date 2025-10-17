/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#ifndef RTC_BASE_EXPERIMENTS_FIELD_TRIAL_UNITS_H_
#define RTC_BASE_EXPERIMENTS_FIELD_TRIAL_UNITS_H_

#include "absl/strings/string_view.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "rtc_base/experiments/field_trial_parser.h"

namespace webrtc {

template <>
std::optional<DataRate> ParseTypedParameter<DataRate>(absl::string_view str);
template <>
std::optional<DataSize> ParseTypedParameter<DataSize>(absl::string_view str);
template <>
std::optional<TimeDelta> ParseTypedParameter<TimeDelta>(absl::string_view str);

extern template class FieldTrialParameter<DataRate>;
extern template class FieldTrialParameter<DataSize>;
extern template class FieldTrialParameter<TimeDelta>;

extern template class FieldTrialConstrained<DataRate>;
extern template class FieldTrialConstrained<DataSize>;
extern template class FieldTrialConstrained<TimeDelta>;

extern template class FieldTrialOptional<DataRate>;
extern template class FieldTrialOptional<DataSize>;
extern template class FieldTrialOptional<TimeDelta>;
}  // namespace webrtc

#endif  // RTC_BASE_EXPERIMENTS_FIELD_TRIAL_UNITS_H_
