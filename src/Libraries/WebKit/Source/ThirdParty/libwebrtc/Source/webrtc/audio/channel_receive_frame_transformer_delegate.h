/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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
#ifndef AUDIO_CHANNEL_RECEIVE_FRAME_TRANSFORMER_DELEGATE_H_
#define AUDIO_CHANNEL_RECEIVE_FRAME_TRANSFORMER_DELEGATE_H_

#include <memory>
#include <string>

#include "api/frame_transformer_interface.h"
#include "api/rtp_headers.h"
#include "api/sequence_checker.h"
#include "api/task_queue/task_queue_base.h"
#include "api/units/timestamp.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread.h"

namespace webrtc {

// Delegates calls to FrameTransformerInterface to transform frames, and to
// ChannelReceive to receive the transformed frames using the
// `receive_frame_callback_` on the `channel_receive_thread_`.
class ChannelReceiveFrameTransformerDelegate : public TransformedFrameCallback {
 public:
  using ReceiveFrameCallback =
      std::function<void(rtc::ArrayView<const uint8_t> packet,
                         const RTPHeader& header,
                         Timestamp receive_time)>;
  ChannelReceiveFrameTransformerDelegate(
      ReceiveFrameCallback receive_frame_callback,
      rtc::scoped_refptr<FrameTransformerInterface> frame_transformer,
      TaskQueueBase* channel_receive_thread);

  // Registers `this` as callback for `frame_transformer_`, to get the
  // transformed frames.
  void Init();

  // Unregisters and releases the `frame_transformer_` reference, and resets
  // `receive_frame_callback_` on `channel_receive_thread_`. Called from
  // ChannelReceive destructor to prevent running the callback on a dangling
  // channel.
  void Reset();

  // Delegates the call to FrameTransformerInterface::Transform, to transform
  // the frame asynchronously.
  void Transform(rtc::ArrayView<const uint8_t> packet,
                 const RTPHeader& header,
                 uint32_t ssrc,
                 const std::string& codec_mime_type,
                 Timestamp receive_time);

  // Implements TransformedFrameCallback. Can be called on any thread.
  void OnTransformedFrame(
      std::unique_ptr<TransformableFrameInterface> frame) override;

  void StartShortCircuiting() override;

  // Delegates the call to ChannelReceive::OnReceivedPayloadData on the
  // `channel_receive_thread_`, by calling `receive_frame_callback_`.
  void ReceiveFrame(std::unique_ptr<TransformableFrameInterface> frame) const;

  rtc::scoped_refptr<FrameTransformerInterface> FrameTransformer();

 protected:
  ~ChannelReceiveFrameTransformerDelegate() override = default;

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker sequence_checker_;
  ReceiveFrameCallback receive_frame_callback_
      RTC_GUARDED_BY(sequence_checker_);
  rtc::scoped_refptr<FrameTransformerInterface> frame_transformer_
      RTC_GUARDED_BY(sequence_checker_);
  TaskQueueBase* const channel_receive_thread_;
  bool short_circuit_ RTC_GUARDED_BY(sequence_checker_) = false;
};

std::unique_ptr<TransformableAudioFrameInterface> CloneReceiverAudioFrame(
    TransformableAudioFrameInterface* original);

}  // namespace webrtc
#endif  // AUDIO_CHANNEL_RECEIVE_FRAME_TRANSFORMER_DELEGATE_H_
