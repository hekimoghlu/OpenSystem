/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
#include "modules/audio_processing/logging/apm_data_dumper.h"

#include "absl/strings/string_view.h"
#include "rtc_base/strings/string_builder.h"

// Check to verify that the define is properly set.
#if !defined(WEBRTC_APM_DEBUG_DUMP) || \
    (WEBRTC_APM_DEBUG_DUMP != 0 && WEBRTC_APM_DEBUG_DUMP != 1)
#error "Set WEBRTC_APM_DEBUG_DUMP to either 0 or 1"
#endif

namespace webrtc {
namespace {

#if WEBRTC_APM_DEBUG_DUMP == 1

#if defined(WEBRTC_WIN)
constexpr char kPathDelimiter = '\\';
#else
constexpr char kPathDelimiter = '/';
#endif

std::string FormFileName(absl::string_view output_dir,
                         absl::string_view name,
                         int instance_index,
                         int reinit_index,
                         absl::string_view suffix) {
  char buf[1024];
  rtc::SimpleStringBuilder ss(buf);
  if (!output_dir.empty()) {
    ss << output_dir;
    if (output_dir.back() != kPathDelimiter) {
      ss << kPathDelimiter;
    }
  }
  ss << name << "_" << instance_index << "-" << reinit_index << suffix;
  return ss.str();
}
#endif

}  // namespace

#if WEBRTC_APM_DEBUG_DUMP == 1
ApmDataDumper::ApmDataDumper(int instance_index)
    : instance_index_(instance_index) {}
#else
ApmDataDumper::ApmDataDumper(int /* instance_index */) {}
#endif

ApmDataDumper::~ApmDataDumper() = default;

#if WEBRTC_APM_DEBUG_DUMP == 1
bool ApmDataDumper::recording_activated_ = false;
std::optional<int> ApmDataDumper::dump_set_to_use_;
char ApmDataDumper::output_dir_[] = "";

FILE* ApmDataDumper::GetRawFile(absl::string_view name) {
  std::string filename = FormFileName(output_dir_, name, instance_index_,
                                      recording_set_index_, ".dat");
  auto& f = raw_files_[filename];
  if (!f) {
    f.reset(fopen(filename.c_str(), "wb"));
    RTC_CHECK(f.get()) << "Cannot write to " << filename << ".";
  }
  return f.get();
}

WavWriter* ApmDataDumper::GetWavFile(absl::string_view name,
                                     int sample_rate_hz,
                                     int num_channels,
                                     WavFile::SampleFormat format) {
  std::string filename = FormFileName(output_dir_, name, instance_index_,
                                      recording_set_index_, ".wav");
  auto& f = wav_files_[filename];
  if (!f) {
    f.reset(
        new WavWriter(filename.c_str(), sample_rate_hz, num_channels, format));
  }
  return f.get();
}
#endif

}  // namespace webrtc
