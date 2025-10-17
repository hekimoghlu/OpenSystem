/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#include "modules/async_audio_processing/async_audio_processing.h"

#include <utility>

#include "api/audio/audio_frame.h"
#include "api/task_queue/task_queue_factory.h"
#include "rtc_base/checks.h"

namespace webrtc {

AsyncAudioProcessing::Factory::~Factory() = default;
AsyncAudioProcessing::Factory::Factory(AudioFrameProcessor& frame_processor,
                                       TaskQueueFactory& task_queue_factory)
    : frame_processor_(frame_processor),
      task_queue_factory_(task_queue_factory) {}

AsyncAudioProcessing::Factory::Factory(
    std::unique_ptr<AudioFrameProcessor> frame_processor,
    TaskQueueFactory& task_queue_factory)
    : frame_processor_(*frame_processor),
      owned_frame_processor_(std::move(frame_processor)),
      task_queue_factory_(task_queue_factory) {}

std::unique_ptr<AsyncAudioProcessing>
AsyncAudioProcessing::Factory::CreateAsyncAudioProcessing(
    AudioFrameProcessor::OnAudioFrameCallback on_frame_processed_callback) {
  if (owned_frame_processor_) {
    return std::make_unique<AsyncAudioProcessing>(
        std::move(owned_frame_processor_), task_queue_factory_,
        std::move(on_frame_processed_callback));
  } else {
    return std::make_unique<AsyncAudioProcessing>(
        frame_processor_, task_queue_factory_,
        std::move(on_frame_processed_callback));
  }
}

AsyncAudioProcessing::~AsyncAudioProcessing() {
  if (owned_frame_processor_) {
    owned_frame_processor_->SetSink(nullptr);
  } else {
    frame_processor_.SetSink(nullptr);
  }
}

AsyncAudioProcessing::AsyncAudioProcessing(
    AudioFrameProcessor& frame_processor,
    TaskQueueFactory& task_queue_factory,
    AudioFrameProcessor::OnAudioFrameCallback on_frame_processed_callback)
    : on_frame_processed_callback_(std::move(on_frame_processed_callback)),
      frame_processor_(frame_processor),
      task_queue_(task_queue_factory.CreateTaskQueue(
          "AsyncAudioProcessing",
          TaskQueueFactory::Priority::NORMAL)) {
  frame_processor_.SetSink([this](std::unique_ptr<AudioFrame> frame) {
    task_queue_->PostTask([this, frame = std::move(frame)]() mutable {
      on_frame_processed_callback_(std::move(frame));
    });
  });
}

AsyncAudioProcessing::AsyncAudioProcessing(
    std::unique_ptr<AudioFrameProcessor> frame_processor,
    TaskQueueFactory& task_queue_factory,
    AudioFrameProcessor::OnAudioFrameCallback on_frame_processed_callback)
    : on_frame_processed_callback_(std::move(on_frame_processed_callback)),
      frame_processor_(*frame_processor),
      owned_frame_processor_(std::move(frame_processor)),
      task_queue_(task_queue_factory.CreateTaskQueue(
          "AsyncAudioProcessing",
          TaskQueueFactory::Priority::NORMAL)) {
  owned_frame_processor_->SetSink([this](std::unique_ptr<AudioFrame> frame) {
    task_queue_->PostTask([this, frame = std::move(frame)]() mutable {
      on_frame_processed_callback_(std::move(frame));
    });
  });
}

void AsyncAudioProcessing::Process(std::unique_ptr<AudioFrame> frame) {
  if (owned_frame_processor_) {
    task_queue_->PostTask([this, frame = std::move(frame)]() mutable {
      owned_frame_processor_->Process(std::move(frame));
    });
  } else {
    task_queue_->PostTask([this, frame = std::move(frame)]() mutable {
      frame_processor_.Process(std::move(frame));
    });
  }
}

}  // namespace webrtc
