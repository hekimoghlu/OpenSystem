/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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
#ifndef MODULES_ASYNC_AUDIO_PROCESSING_ASYNC_AUDIO_PROCESSING_H_
#define MODULES_ASYNC_AUDIO_PROCESSING_ASYNC_AUDIO_PROCESSING_H_

#include <memory>

#include "api/audio/audio_frame_processor.h"
#include "api/task_queue/task_queue_base.h"
#include "rtc_base/ref_count.h"

namespace webrtc {

class AudioFrame;
class TaskQueueFactory;

// Helper class taking care of interactions with AudioFrameProcessor
// in asynchronous manner. Offloads AudioFrameProcessor::Process calls
// to a dedicated task queue. Makes sure that it's always safe for
// AudioFrameProcessor to pass processed frames back to its sink.
class AsyncAudioProcessing final {
 public:
  // Helper class passing AudioFrameProcessor and TaskQueueFactory into
  // AsyncAudioProcessing constructor.
  class Factory : public RefCountInterface {
   public:
    Factory(const Factory&) = delete;
    Factory& operator=(const Factory&) = delete;

    ~Factory();
    Factory(AudioFrameProcessor& frame_processor,
            TaskQueueFactory& task_queue_factory);
    Factory(std::unique_ptr<AudioFrameProcessor> frame_processor,
            TaskQueueFactory& task_queue_factory);

    std::unique_ptr<AsyncAudioProcessing> CreateAsyncAudioProcessing(
        AudioFrameProcessor::OnAudioFrameCallback on_frame_processed_callback);

   private:
    // TODO(bugs.webrtc.org/15111):
    //   Remove 'AudioFrameProcessor& frame_processor_' in favour of
    //   std::unique_ptr in the follow-up.
    //   While transitioning this API from using AudioFrameProcessor& to using
    //   std::unique_ptr<AudioFrameProcessor>, we have two member variable both
    //   referencing the same object. Throughout the lifetime of the Factory
    //   only one of the variables is used, depending on which constructor was
    //   called.
    AudioFrameProcessor& frame_processor_;
    std::unique_ptr<AudioFrameProcessor> owned_frame_processor_;
    TaskQueueFactory& task_queue_factory_;
  };

  AsyncAudioProcessing(const AsyncAudioProcessing&) = delete;
  AsyncAudioProcessing& operator=(const AsyncAudioProcessing&) = delete;

  ~AsyncAudioProcessing();

  // Creates AsyncAudioProcessing which will pass audio frames to
  // `frame_processor` on `task_queue_` and reply with processed frames passed
  // into `on_frame_processed_callback`, which is posted back onto
  // `task_queue_`. `task_queue_` is created using the provided
  // `task_queue_factory`.
  // TODO(bugs.webrtc.org/15111):
  //   Remove this method in favour of the method taking the
  //   unique_ptr<AudioFrameProcessor> in the follow-up.
  AsyncAudioProcessing(
      AudioFrameProcessor& frame_processor,
      TaskQueueFactory& task_queue_factory,
      AudioFrameProcessor::OnAudioFrameCallback on_frame_processed_callback);

  // Creates AsyncAudioProcessing which will pass audio frames to
  // `frame_processor` on `task_queue_` and reply with processed frames passed
  // into `on_frame_processed_callback`, which is posted back onto
  // `task_queue_`. `task_queue_` is created using the provided
  // `task_queue_factory`.
  AsyncAudioProcessing(
      std::unique_ptr<AudioFrameProcessor> frame_processor,
      TaskQueueFactory& task_queue_factory,
      AudioFrameProcessor::OnAudioFrameCallback on_frame_processed_callback);

  // Accepts `frame` for asynchronous processing. Thread-safe.
  void Process(std::unique_ptr<AudioFrame> frame);

 private:
  AudioFrameProcessor::OnAudioFrameCallback on_frame_processed_callback_;
  // TODO(bugs.webrtc.org/15111):
  //   Remove 'AudioFrameProcessor& frame_processor_' in favour of
  //   std::unique_ptr in the follow-up.
  //   While transitioning this API from using AudioFrameProcessor& to using
  //   std::unique_ptr<AudioFrameProcessor>, we have two member variable both
  //   referencing the same object. Throughout the lifetime of the Factory
  //   only one of the variables is used, depending on which constructor was
  //   called.
  AudioFrameProcessor& frame_processor_;
  std::unique_ptr<AudioFrameProcessor> owned_frame_processor_;
  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> task_queue_;
};

}  // namespace webrtc

#endif  // MODULES_ASYNC_AUDIO_PROCESSING_ASYNC_AUDIO_PROCESSING_H_
