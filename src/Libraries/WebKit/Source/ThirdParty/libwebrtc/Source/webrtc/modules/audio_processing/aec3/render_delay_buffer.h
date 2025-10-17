/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_BUFFER_H_

#include <stddef.h>

#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/downsampled_render_buffer.h"
#include "modules/audio_processing/aec3/render_buffer.h"

namespace webrtc {

// Class for buffering the incoming render blocks such that these may be
// extracted with a specified delay.
class RenderDelayBuffer {
 public:
  enum class BufferingEvent {
    kNone,
    kRenderUnderrun,
    kRenderOverrun,
    kApiCallSkew
  };

  static RenderDelayBuffer* Create(const EchoCanceller3Config& config,
                                   int sample_rate_hz,
                                   size_t num_render_channels);
  virtual ~RenderDelayBuffer() = default;

  // Resets the buffer alignment.
  virtual void Reset() = 0;

  // Inserts a block into the buffer.
  virtual BufferingEvent Insert(const Block& block) = 0;

  // Updates the buffers one step based on the specified buffer delay. Returns
  // an enum indicating whether there was a special event that occurred.
  virtual BufferingEvent PrepareCaptureProcessing() = 0;

  // Called on capture blocks where PrepareCaptureProcessing is not called.
  virtual void HandleSkippedCaptureProcessing() = 0;

  // Sets the buffer delay and returns a bool indicating whether the delay
  // changed.
  virtual bool AlignFromDelay(size_t delay) = 0;

  // Sets the buffer delay from the most recently reported external delay.
  virtual void AlignFromExternalDelay() = 0;

  // Gets the buffer delay.
  virtual size_t Delay() const = 0;

  // Gets the buffer delay.
  virtual size_t MaxDelay() const = 0;

  // Returns the render buffer for the echo remover.
  virtual RenderBuffer* GetRenderBuffer() = 0;

  // Returns the downsampled render buffer.
  virtual const DownsampledRenderBuffer& GetDownsampledRenderBuffer() const = 0;

  // Returns the maximum non calusal offset that can occur in the delay buffer.
  static int DelayEstimatorOffset(const EchoCanceller3Config& config);

  // Provides an optional external estimate of the audio buffer delay.
  virtual void SetAudioBufferDelay(int delay_ms) = 0;

  // Returns whether an external delay estimate has been reported via
  // SetAudioBufferDelay.
  virtual bool HasReceivedBufferDelay() = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_BUFFER_H_
