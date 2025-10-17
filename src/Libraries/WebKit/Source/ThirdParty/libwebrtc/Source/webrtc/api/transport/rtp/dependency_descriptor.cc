/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#include "api/transport/rtp/dependency_descriptor.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

namespace webrtc {

constexpr int DependencyDescriptor::kMaxSpatialIds;
constexpr int DependencyDescriptor::kMaxTemporalIds;
constexpr int DependencyDescriptor::kMaxTemplates;
constexpr int DependencyDescriptor::kMaxDecodeTargets;

namespace webrtc_impl {

absl::InlinedVector<DecodeTargetIndication, 10> StringToDecodeTargetIndications(
    absl::string_view symbols) {
  absl::InlinedVector<DecodeTargetIndication, 10> dtis;
  dtis.reserve(symbols.size());
  for (char symbol : symbols) {
    DecodeTargetIndication indication;
    switch (symbol) {
      case '-':
        indication = DecodeTargetIndication::kNotPresent;
        break;
      case 'D':
        indication = DecodeTargetIndication::kDiscardable;
        break;
      case 'R':
        indication = DecodeTargetIndication::kRequired;
        break;
      case 'S':
        indication = DecodeTargetIndication::kSwitch;
        break;
      default:
        RTC_DCHECK_NOTREACHED();
    }
    dtis.push_back(indication);
  }
  return dtis;
}

}  // namespace webrtc_impl
}  // namespace webrtc
