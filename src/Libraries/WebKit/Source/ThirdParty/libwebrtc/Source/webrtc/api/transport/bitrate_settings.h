/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#ifndef API_TRANSPORT_BITRATE_SETTINGS_H_
#define API_TRANSPORT_BITRATE_SETTINGS_H_

#include <optional>

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Configuration of send bitrate. The `start_bitrate_bps` value is
// used for multiple purposes, both as a prior in the bandwidth
// estimator, and for initial configuration of the encoder. We may
// want to create separate apis for those, and use a smaller struct
// with only the min and max constraints.
struct RTC_EXPORT BitrateSettings {
  BitrateSettings();
  ~BitrateSettings();
  BitrateSettings(const BitrateSettings&);
  // 0 <= min <= start <= max should hold for set parameters.
  std::optional<int> min_bitrate_bps;
  std::optional<int> start_bitrate_bps;
  std::optional<int> max_bitrate_bps;
};

// TODO(srte): BitrateConstraints and BitrateSettings should be merged.
// Both represent the same kind data, but are using different default
// initializer and representation of unset values.
struct BitrateConstraints {
  int min_bitrate_bps = 0;
  int start_bitrate_bps = kDefaultStartBitrateBps;
  int max_bitrate_bps = -1;

 private:
  static constexpr int kDefaultStartBitrateBps = 300000;
};

}  // namespace webrtc

#endif  // API_TRANSPORT_BITRATE_SETTINGS_H_
