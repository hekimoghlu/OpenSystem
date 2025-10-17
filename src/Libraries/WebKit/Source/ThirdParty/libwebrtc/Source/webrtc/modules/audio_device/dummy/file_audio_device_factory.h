/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#ifndef AUDIO_DEVICE_FILE_AUDIO_DEVICE_FACTORY_H_
#define AUDIO_DEVICE_FILE_AUDIO_DEVICE_FACTORY_H_

#include <stdint.h>

#include "absl/strings/string_view.h"

namespace webrtc {

class FileAudioDevice;

// This class is used by audio_device_impl.cc when WebRTC is compiled with
// WEBRTC_DUMMY_FILE_DEVICES. The application must include this file and set the
// filenames to use before the audio device module is initialized. This is
// intended for test tools which use the audio device module.
class FileAudioDeviceFactory {
 public:
  static FileAudioDevice* CreateFileAudioDevice();

  // The input file must be a readable 48k stereo raw file. The output
  // file must be writable. The strings will be copied.
  static void SetFilenamesToUse(absl::string_view inputAudioFilename,
                                absl::string_view outputAudioFilename);

 private:
  enum : uint32_t { MAX_FILENAME_LEN = 512 };
  static bool _isConfigured;
  static char _inputAudioFilename[MAX_FILENAME_LEN];
  static char _outputAudioFilename[MAX_FILENAME_LEN];
};

}  // namespace webrtc

#endif  // AUDIO_DEVICE_FILE_AUDIO_DEVICE_FACTORY_H_
