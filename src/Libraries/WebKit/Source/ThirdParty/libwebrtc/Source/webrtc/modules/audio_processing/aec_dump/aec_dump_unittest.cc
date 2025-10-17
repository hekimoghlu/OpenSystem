/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
#include <array>
#include <utility>

#include "modules/audio_processing/aec_dump/aec_dump_factory.h"
#include "rtc_base/task_queue_for_test.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {

TEST(AecDumper, APICallsDoNotCrash) {
  // Note order of initialization: Task queue has to be initialized
  // before AecDump.
  webrtc::TaskQueueForTest file_writer_queue("file_writer_queue");

  const std::string filename =
      webrtc::test::TempFilename(webrtc::test::OutputPath(), "aec_dump");

  {
    std::unique_ptr<webrtc::AecDump> aec_dump =
        webrtc::AecDumpFactory::Create(filename, -1, file_writer_queue.Get());

    constexpr int kNumChannels = 1;
    constexpr int kNumSamplesPerChannel = 160;
    std::array<int16_t, kNumSamplesPerChannel * kNumChannels> frame;
    frame.fill(0.f);
    aec_dump->WriteRenderStreamMessage(frame.data(), kNumChannels,
                                       kNumSamplesPerChannel);

    aec_dump->AddCaptureStreamInput(frame.data(), kNumChannels,
                                    kNumSamplesPerChannel);
    aec_dump->AddCaptureStreamOutput(frame.data(), kNumChannels,
                                     kNumSamplesPerChannel);

    aec_dump->WriteCaptureStreamMessage();

    webrtc::InternalAPMConfig apm_config;
    aec_dump->WriteConfig(apm_config);

    webrtc::ProcessingConfig api_format;
    constexpr int64_t kTimeNowMs = 123456789ll;
    aec_dump->WriteInitMessage(api_format, kTimeNowMs);
  }
  // Remove file after the AecDump d-tor has finished.
  ASSERT_EQ(0, remove(filename.c_str()));
}

TEST(AecDumper, WriteToFile) {
  webrtc::TaskQueueForTest file_writer_queue("file_writer_queue");

  const std::string filename =
      webrtc::test::TempFilename(webrtc::test::OutputPath(), "aec_dump");

  {
    std::unique_ptr<webrtc::AecDump> aec_dump =
        webrtc::AecDumpFactory::Create(filename, -1, file_writer_queue.Get());

    constexpr int kNumChannels = 1;
    constexpr int kNumSamplesPerChannel = 160;
    std::array<int16_t, kNumSamplesPerChannel * kNumChannels> frame;
    frame.fill(0.f);

    aec_dump->WriteRenderStreamMessage(frame.data(), kNumChannels,
                                       kNumSamplesPerChannel);
  }

  // Verify the file has been written after the AecDump d-tor has
  // finished.
  FILE* fid = fopen(filename.c_str(), "r");
  ASSERT_TRUE(fid != NULL);

  // Clean it up.
  ASSERT_EQ(0, fclose(fid));
  ASSERT_EQ(0, remove(filename.c_str()));
}

}  // namespace webrtc
