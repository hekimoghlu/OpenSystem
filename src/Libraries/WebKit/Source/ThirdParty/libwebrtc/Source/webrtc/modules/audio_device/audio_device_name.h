/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#ifndef MODULES_AUDIO_DEVICE_AUDIO_DEVICE_NAME_H_
#define MODULES_AUDIO_DEVICE_AUDIO_DEVICE_NAME_H_

#include <deque>
#include <string>

#include "absl/strings/string_view.h"

namespace webrtc {

struct AudioDeviceName {
  // Represents a default device. Note that, on Windows there are two different
  // types of default devices (Default and Default Communication). They can
  // either be two different physical devices or be two different roles for one
  // single device. Hence, this id must be combined with a "role parameter" on
  // Windows to uniquely identify a default device.
  static const char kDefaultDeviceId[];

  AudioDeviceName() = default;
  AudioDeviceName(absl::string_view device_name, absl::string_view unique_id);

  ~AudioDeviceName() = default;

  // Support copy and move.
  AudioDeviceName(const AudioDeviceName& other) = default;
  AudioDeviceName(AudioDeviceName&&) = default;
  AudioDeviceName& operator=(const AudioDeviceName&) = default;
  AudioDeviceName& operator=(AudioDeviceName&&) = default;

  bool IsValid();

  std::string device_name;  // Friendly name of the device.
  std::string unique_id;    // Unique identifier for the device.
};

typedef std::deque<AudioDeviceName> AudioDeviceNames;

}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_AUDIO_DEVICE_NAME_H_
