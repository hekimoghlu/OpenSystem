/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_SCREENCAST_STREAM_UTILS_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_SCREENCAST_STREAM_UTILS_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "rtc_base/string_encode.h"

struct spa_pod;
struct spa_pod_builder;
struct spa_rectangle;
struct spa_fraction;

namespace webrtc {

struct PipeWireVersion {
  static PipeWireVersion Parse(const absl::string_view& version);

  // Returns whether current version is newer or same as required version
  bool operator>=(const PipeWireVersion& other);
  // Returns whether current version is older or same as required version
  bool operator<=(const PipeWireVersion& other);

  int major = 0;
  int minor = 0;
  int micro = 0;
};

// Returns a spa_pod used to build PipeWire stream format using given
// arguments. Modifiers are optional value and when present they will be
// used with SPA_POD_PROP_FLAG_MANDATORY and SPA_POD_PROP_FLAG_DONT_FIXATE
// flags.
spa_pod* BuildFormat(spa_pod_builder* builder,
                     uint32_t format,
                     const std::vector<uint64_t>& modifiers,
                     const struct spa_rectangle* resolution,
                     const struct spa_fraction* frame_rate);

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_SCREENCAST_STREAM_UTILS_H_
