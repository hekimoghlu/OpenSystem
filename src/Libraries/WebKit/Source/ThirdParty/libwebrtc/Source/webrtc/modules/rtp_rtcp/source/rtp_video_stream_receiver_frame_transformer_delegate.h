/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_VIDEO_STREAM_RECEIVER_FRAME_TRANSFORMER_DELEGATE_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_VIDEO_STREAM_RECEIVER_FRAME_TRANSFORMER_DELEGATE_H_

#include <memory>

#include "api/frame_transformer_interface.h"
#include "api/sequence_checker.h"
#include "modules/rtp_rtcp/source/frame_object.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

// Called back by RtpVideoStreamReceiverFrameTransformerDelegate on the network
// thread after transformation.
class RtpVideoFrameReceiver {
 public:
  virtual void ManageFrame(std::unique_ptr<RtpFrameObject> frame) = 0;

 protected:
  virtual ~RtpVideoFrameReceiver() = default;
};

// Delegates calls to FrameTransformerInterface to transform frames, and to
// RtpVideoStreamReceiver to manage transformed frames on the `network_thread_`.
class RtpVideoStreamReceiverFrameTransformerDelegate
    : public TransformedFrameCallback {
 public:
  RtpVideoStreamReceiverFrameTransformerDelegate(
      RtpVideoFrameReceiver* receiver,
      Clock* clock,
      rtc::scoped_refptr<FrameTransformerInterface> frame_transformer,
      rtc::Thread* network_thread,
      uint32_t ssrc);

  void Init();
  void Reset();

  // Delegates the call to FrameTransformerInterface::TransformFrame.
  void TransformFrame(std::unique_ptr<RtpFrameObject> frame);

  // Implements TransformedFrameCallback. Can be called on any thread. Posts
  // the transformed frame to be managed on the `network_thread_`.
  void OnTransformedFrame(
      std::unique_ptr<TransformableFrameInterface> frame) override;

  void StartShortCircuiting() override;

  // Delegates the call to RtpVideoFrameReceiver::ManageFrame on the
  // `network_thread_`.
  void ManageFrame(std::unique_ptr<TransformableFrameInterface> frame);

 protected:
  ~RtpVideoStreamReceiverFrameTransformerDelegate() override = default;

 private:
  void StartShortCircuitingOnNetworkSequence();

  RTC_NO_UNIQUE_ADDRESS SequenceChecker network_sequence_checker_;
  RtpVideoFrameReceiver* receiver_ RTC_GUARDED_BY(network_sequence_checker_);
  rtc::scoped_refptr<FrameTransformerInterface> frame_transformer_
      RTC_GUARDED_BY(network_sequence_checker_);
  rtc::Thread* const network_thread_;
  const uint32_t ssrc_;
  Clock* const clock_;
  bool short_circuit_ RTC_GUARDED_BY(network_sequence_checker_) = false;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_VIDEO_STREAM_RECEIVER_FRAME_TRANSFORMER_DELEGATE_H_
