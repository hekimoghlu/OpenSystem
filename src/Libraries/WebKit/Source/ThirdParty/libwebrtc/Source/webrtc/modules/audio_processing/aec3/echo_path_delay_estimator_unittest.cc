/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#include "modules/audio_processing/aec3/echo_path_delay_estimator.h"

#include <algorithm>
#include <string>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/render_delay_buffer.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "modules/audio_processing/test/echo_canceller_test_tools.h"
#include "rtc_base/random.h"
#include "rtc_base/strings/string_builder.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

std::string ProduceDebugText(size_t delay, size_t down_sampling_factor) {
  rtc::StringBuilder ss;
  ss << "Delay: " << delay;
  ss << ", Down sampling factor: " << down_sampling_factor;
  return ss.Release();
}

}  // namespace

class EchoPathDelayEstimatorMultiChannel
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t>> {};

INSTANTIATE_TEST_SUITE_P(MultiChannel,
                         EchoPathDelayEstimatorMultiChannel,
                         ::testing::Combine(::testing::Values(1, 2, 3, 6, 8),
                                            ::testing::Values(1, 2, 4)));

// Verifies that the basic API calls work.
TEST_P(EchoPathDelayEstimatorMultiChannel, BasicApiCalls) {
  const size_t num_render_channels = std::get<0>(GetParam());
  const size_t num_capture_channels = std::get<1>(GetParam());
  constexpr int kSampleRateHz = 48000;
  constexpr size_t kNumBands = NumBandsForRate(kSampleRateHz);
  ApmDataDumper data_dumper(0);
  EchoCanceller3Config config;
  std::unique_ptr<RenderDelayBuffer> render_delay_buffer(
      RenderDelayBuffer::Create(config, kSampleRateHz, num_render_channels));
  EchoPathDelayEstimator estimator(&data_dumper, config, num_capture_channels);
  Block render(kNumBands, num_render_channels);
  Block capture(/*num_bands=*/1, num_capture_channels);
  for (size_t k = 0; k < 100; ++k) {
    render_delay_buffer->Insert(render);
    estimator.EstimateDelay(render_delay_buffer->GetDownsampledRenderBuffer(),
                            capture);
  }
}

// Verifies that the delay estimator produces correct delay for artificially
// delayed signals.
TEST(EchoPathDelayEstimator, DelayEstimation) {
  constexpr size_t kNumRenderChannels = 1;
  constexpr size_t kNumCaptureChannels = 1;
  constexpr int kSampleRateHz = 48000;
  constexpr size_t kNumBands = NumBandsForRate(kSampleRateHz);
  Random random_generator(42U);
  Block render(kNumBands, kNumRenderChannels);
  Block capture(/*num_bands=*/1, kNumCaptureChannels);
  ApmDataDumper data_dumper(0);
  constexpr size_t kDownSamplingFactors[] = {2, 4, 8};
  for (auto down_sampling_factor : kDownSamplingFactors) {
    EchoCanceller3Config config;
    config.delay.delay_headroom_samples = 0;
    config.delay.down_sampling_factor = down_sampling_factor;
    config.delay.num_filters = 10;
    for (size_t delay_samples : {30, 64, 150, 200, 800, 4000}) {
      SCOPED_TRACE(ProduceDebugText(delay_samples, down_sampling_factor));
      std::unique_ptr<RenderDelayBuffer> render_delay_buffer(
          RenderDelayBuffer::Create(config, kSampleRateHz, kNumRenderChannels));
      DelayBuffer<float> signal_delay_buffer(delay_samples);
      EchoPathDelayEstimator estimator(&data_dumper, config,
                                       kNumCaptureChannels);

      std::optional<DelayEstimate> estimated_delay_samples;
      for (size_t k = 0; k < (500 + (delay_samples) / kBlockSize); ++k) {
        RandomizeSampleVector(&random_generator,
                              render.View(/*band=*/0, /*channel=*/0));
        signal_delay_buffer.Delay(render.View(/*band=*/0, /*channel=*/0),
                                  capture.View(/*band=*/0, /*channel=*/0));
        render_delay_buffer->Insert(render);

        if (k == 0) {
          render_delay_buffer->Reset();
        }

        render_delay_buffer->PrepareCaptureProcessing();

        auto estimate = estimator.EstimateDelay(
            render_delay_buffer->GetDownsampledRenderBuffer(), capture);

        if (estimate) {
          estimated_delay_samples = estimate;
        }
      }

      if (estimated_delay_samples) {
        // Allow estimated delay to be off by a block as internally the delay is
        // quantized with an error up to a block.
        size_t delay_ds = delay_samples / down_sampling_factor;
        size_t estimated_delay_ds =
            estimated_delay_samples->delay / down_sampling_factor;
        EXPECT_NEAR(delay_ds, estimated_delay_ds,
                    kBlockSize / down_sampling_factor);
      } else {
        ADD_FAILURE();
      }
    }
  }
}

// Verifies that the delay estimator does not produce delay estimates for render
// signals of low level.
TEST(EchoPathDelayEstimator, NoDelayEstimatesForLowLevelRenderSignals) {
  constexpr size_t kNumRenderChannels = 1;
  constexpr size_t kNumCaptureChannels = 1;
  constexpr int kSampleRateHz = 48000;
  constexpr size_t kNumBands = NumBandsForRate(kSampleRateHz);
  Random random_generator(42U);
  EchoCanceller3Config config;
  Block render(kNumBands, kNumRenderChannels);
  Block capture(/*num_bands=*/1, kNumCaptureChannels);
  ApmDataDumper data_dumper(0);
  EchoPathDelayEstimator estimator(&data_dumper, config, kNumCaptureChannels);
  std::unique_ptr<RenderDelayBuffer> render_delay_buffer(
      RenderDelayBuffer::Create(EchoCanceller3Config(), kSampleRateHz,
                                kNumRenderChannels));
  for (size_t k = 0; k < 100; ++k) {
    RandomizeSampleVector(&random_generator,
                          render.View(/*band=*/0, /*channel=*/0));
    for (auto& render_k : render.View(/*band=*/0, /*channel=*/0)) {
      render_k *= 100.f / 32767.f;
    }
    std::copy(render.begin(/*band=*/0, /*channel=*/0),
              render.end(/*band=*/0, /*channel=*/0),
              capture.begin(/*band*/ 0, /*channel=*/0));
    render_delay_buffer->Insert(render);
    render_delay_buffer->PrepareCaptureProcessing();
    EXPECT_FALSE(estimator.EstimateDelay(
        render_delay_buffer->GetDownsampledRenderBuffer(), capture));
  }
}

#if RTC_DCHECK_IS_ON && GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)

// Verifies the check for the render blocksize.
// TODO(peah): Re-enable the test once the issue with memory leaks during DEATH
// tests on test bots has been fixed.
TEST(EchoPathDelayEstimatorDeathTest, DISABLED_WrongRenderBlockSize) {
  ApmDataDumper data_dumper(0);
  EchoCanceller3Config config;
  EchoPathDelayEstimator estimator(&data_dumper, config, 1);
  std::unique_ptr<RenderDelayBuffer> render_delay_buffer(
      RenderDelayBuffer::Create(config, 48000, 1));
  Block capture(/*num_bands=*/1, /*num_channels=*/1);
  EXPECT_DEATH(estimator.EstimateDelay(
                   render_delay_buffer->GetDownsampledRenderBuffer(), capture),
               "");
}

// Verifies the check for non-null data dumper.
TEST(EchoPathDelayEstimatorDeathTest, NullDataDumper) {
  EXPECT_DEATH(EchoPathDelayEstimator(nullptr, EchoCanceller3Config(), 1), "");
}

#endif

}  // namespace webrtc
