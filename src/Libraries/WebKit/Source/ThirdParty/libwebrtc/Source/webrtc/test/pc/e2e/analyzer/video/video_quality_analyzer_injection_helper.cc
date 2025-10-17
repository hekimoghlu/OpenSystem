/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#include "test/pc/e2e/analyzer/video/video_quality_analyzer_injection_helper.h"

#include <stdio.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/test/pclf/media_configuration.h"
#include "api/video/i420_buffer.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/strings/string_builder.h"
#include "system_wrappers/include/clock.h"
#include "test/pc/e2e/analyzer/video/analyzing_video_sink.h"
#include "test/pc/e2e/analyzer/video/quality_analyzing_video_decoder.h"
#include "test/pc/e2e/analyzer/video/quality_analyzing_video_encoder.h"
#include "test/pc/e2e/analyzer/video/simulcast_dummy_buffer_helper.h"
#include "test/pc/e2e/analyzer/video/video_dumping.h"
#include "test/testsupport/fixed_fps_video_frame_writer_adapter.h"
#include "test/video_renderer.h"

namespace webrtc {
namespace webrtc_pc_e2e {
namespace {

using webrtc::webrtc_pc_e2e::VideoConfig;
using EmulatedSFUConfigMap =
    ::webrtc::webrtc_pc_e2e::QualityAnalyzingVideoEncoder::EmulatedSFUConfigMap;

class AnalyzingFramePreprocessor
    : public test::TestVideoCapturer::FramePreprocessor {
 public:
  AnalyzingFramePreprocessor(
      absl::string_view peer_name,
      absl::string_view stream_label,
      VideoQualityAnalyzerInterface* analyzer,
      std::vector<std::unique_ptr<rtc::VideoSinkInterface<VideoFrame>>> sinks)
      : peer_name_(peer_name),
        stream_label_(stream_label),
        analyzer_(analyzer),
        sinks_(std::move(sinks)) {}
  ~AnalyzingFramePreprocessor() override = default;

  VideoFrame Preprocess(const VideoFrame& source_frame) override {
    // Copy VideoFrame to be able to set id on it.
    VideoFrame frame = source_frame;
    uint16_t frame_id =
        analyzer_->OnFrameCaptured(peer_name_, stream_label_, frame);
    frame.set_id(frame_id);

    for (auto& sink : sinks_) {
      sink->OnFrame(frame);
    }
    return frame;
  }

 private:
  const std::string peer_name_;
  const std::string stream_label_;
  VideoQualityAnalyzerInterface* const analyzer_;
  const std::vector<std::unique_ptr<rtc::VideoSinkInterface<VideoFrame>>>
      sinks_;
};

}  // namespace

VideoQualityAnalyzerInjectionHelper::VideoQualityAnalyzerInjectionHelper(
    Clock* clock,
    std::unique_ptr<VideoQualityAnalyzerInterface> analyzer,
    EncodedImageDataInjector* injector,
    EncodedImageDataExtractor* extractor)
    : clock_(clock),
      analyzer_(std::move(analyzer)),
      injector_(injector),
      extractor_(extractor) {
  RTC_DCHECK(clock_);
  RTC_DCHECK(injector_);
  RTC_DCHECK(extractor_);
}
VideoQualityAnalyzerInjectionHelper::~VideoQualityAnalyzerInjectionHelper() =
    default;

std::unique_ptr<VideoEncoderFactory>
VideoQualityAnalyzerInjectionHelper::WrapVideoEncoderFactory(
    absl::string_view peer_name,
    std::unique_ptr<VideoEncoderFactory> delegate,
    double bitrate_multiplier,
    EmulatedSFUConfigMap stream_to_sfu_config) const {
  return std::make_unique<QualityAnalyzingVideoEncoderFactory>(
      peer_name, std::move(delegate), bitrate_multiplier,
      std::move(stream_to_sfu_config), injector_, analyzer_.get());
}

std::unique_ptr<VideoDecoderFactory>
VideoQualityAnalyzerInjectionHelper::WrapVideoDecoderFactory(
    absl::string_view peer_name,
    std::unique_ptr<VideoDecoderFactory> delegate) const {
  return std::make_unique<QualityAnalyzingVideoDecoderFactory>(
      peer_name, std::move(delegate), extractor_, analyzer_.get());
}

std::unique_ptr<test::TestVideoCapturer::FramePreprocessor>
VideoQualityAnalyzerInjectionHelper::CreateFramePreprocessor(
    absl::string_view peer_name,
    const VideoConfig& config) {
  std::vector<std::unique_ptr<rtc::VideoSinkInterface<VideoFrame>>> sinks;
  if (config.input_dump_options.has_value()) {
    std::unique_ptr<test::VideoFrameWriter> writer =
        config.input_dump_options->CreateInputDumpVideoFrameWriter(
            *config.stream_label, config.GetResolution());
    sinks.push_back(std::make_unique<VideoWriter>(
        writer.get(), config.input_dump_options->sampling_modulo()));
    video_writers_.push_back(std::move(writer));
  }
  if (config.show_on_screen) {
    sinks.push_back(absl::WrapUnique(
        test::VideoRenderer::Create((*config.stream_label + "-capture").c_str(),
                                    config.width, config.height)));
  }
  sinks_helper_.AddConfig(peer_name, config);
  return std::make_unique<AnalyzingFramePreprocessor>(
      peer_name, std::move(*config.stream_label), analyzer_.get(),
      std::move(sinks));
}

std::unique_ptr<AnalyzingVideoSink>
VideoQualityAnalyzerInjectionHelper::CreateVideoSink(
    absl::string_view peer_name,
    const VideoSubscription& subscription,
    bool report_infra_metrics) {
  return std::make_unique<AnalyzingVideoSink>(peer_name, clock_, *analyzer_,
                                              sinks_helper_, subscription,
                                              report_infra_metrics);
}

void VideoQualityAnalyzerInjectionHelper::Start(
    std::string test_case_name,
    rtc::ArrayView<const std::string> peer_names,
    int max_threads_count) {
  analyzer_->Start(std::move(test_case_name), peer_names, max_threads_count);
  extractor_->Start(peer_names.size());
}

void VideoQualityAnalyzerInjectionHelper::RegisterParticipantInCall(
    absl::string_view peer_name) {
  analyzer_->RegisterParticipantInCall(peer_name);
  extractor_->AddParticipantInCall();
}

void VideoQualityAnalyzerInjectionHelper::UnregisterParticipantInCall(
    absl::string_view peer_name) {
  analyzer_->UnregisterParticipantInCall(peer_name);
  extractor_->RemoveParticipantInCall();
}

void VideoQualityAnalyzerInjectionHelper::OnStatsReports(
    absl::string_view pc_label,
    const rtc::scoped_refptr<const RTCStatsReport>& report) {
  analyzer_->OnStatsReports(pc_label, report);
}

void VideoQualityAnalyzerInjectionHelper::Stop() {
  analyzer_->Stop();
  for (const auto& video_writer : video_writers_) {
    video_writer->Close();
  }
  video_writers_.clear();
  sinks_helper_.Clear();
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
