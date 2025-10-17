/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#include "modules/audio_device/dummy/file_audio_device_factory.h"

#include <stdio.h>

#include <cstdlib>

#include "absl/strings/string_view.h"
#include "modules/audio_device/dummy/file_audio_device.h"
#include "rtc_base/logging.h"
#include "rtc_base/string_utils.h"

namespace webrtc {

bool FileAudioDeviceFactory::_isConfigured = false;
char FileAudioDeviceFactory::_inputAudioFilename[MAX_FILENAME_LEN] = "";
char FileAudioDeviceFactory::_outputAudioFilename[MAX_FILENAME_LEN] = "";

FileAudioDevice* FileAudioDeviceFactory::CreateFileAudioDevice() {
  // Bail out here if the files haven't been set explicitly.
  // audio_device_impl.cc should then fall back to dummy audio.
  if (!_isConfigured) {
    RTC_LOG(LS_WARNING)
        << "WebRTC configured with WEBRTC_DUMMY_FILE_DEVICES but "
           "no device files supplied. Will fall back to dummy "
           "audio.";

    return nullptr;
  }
  return new FileAudioDevice(_inputAudioFilename, _outputAudioFilename);
}

void FileAudioDeviceFactory::SetFilenamesToUse(
    [[maybe_unused]] absl::string_view inputAudioFilename,
    [[maybe_unused]] absl::string_view outputAudioFilename) {
#ifdef WEBRTC_DUMMY_FILE_DEVICES
  RTC_DCHECK_LT(inputAudioFilename.size(), MAX_FILENAME_LEN);
  RTC_DCHECK_LT(outputAudioFilename.size(), MAX_FILENAME_LEN);

  // Copy the strings since we don't know the lifetime of the input pointers.
  rtc::strcpyn(_inputAudioFilename, MAX_FILENAME_LEN, inputAudioFilename);
  rtc::strcpyn(_outputAudioFilename, MAX_FILENAME_LEN, outputAudioFilename);
  _isConfigured = true;
#else
  // Sanity: must be compiled with the right define to run this.
  printf(
      "Trying to use dummy file devices, but is not compiled "
      "with WEBRTC_DUMMY_FILE_DEVICES. Bailing out.\n");
  std::exit(1);
#endif
}

}  // namespace webrtc
