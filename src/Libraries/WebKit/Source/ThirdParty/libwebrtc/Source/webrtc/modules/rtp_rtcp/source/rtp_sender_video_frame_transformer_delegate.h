/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_SENDER_VIDEO_FRAME_TRANSFORMER_DELEGATE_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_SENDER_VIDEO_FRAME_TRANSFORMER_DELEGATE_H_

#include <memory>
#include <vector>

#include "api/frame_transformer_interface.h"
#include "api/scoped_refptr.h"
#include "api/sequence_checker.h"
#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "api/video/encoded_image.h"
#include "api/video/video_layers_allocation.h"
#include "modules/rtp_rtcp/source/rtp_video_header.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// Interface for sending videoframes on an RTP connection, after a transform
// have been applied.
class RTPVideoFrameSenderInterface {
 public:
  virtual bool SendVideo(int payload_type,
                         std::optional<VideoCodecType> codec_type,
                         uint32_t rtp_timestamp,
                         Timestamp capture_time,
                         rtc::ArrayView<const uint8_t> payload,
                         size_t encoder_output_size,
                         RTPVideoHeader video_header,
                         TimeDelta expected_retransmission_time,
                         std::vector<uint32_t> csrcs) = 0;

  virtual void SetVideoStructureAfterTransformation(
      const FrameDependencyStructure* video_structure) = 0;
  virtual void SetVideoLayersAllocationAfterTransformation(
      VideoLayersAllocation allocation) = 0;

 protected:
  virtual ~RTPVideoFrameSenderInterface() = default;
};

// Delegates calls to FrameTransformerInterface to transform frames, and to
// RTPSenderVideo to send the transformed frames. Ensures thread-safe access to
// the sender.
class RTPSenderVideoFrameTransformerDelegate : public TransformedFrameCallback {
 public:
  RTPSenderVideoFrameTransformerDelegate(
      RTPVideoFrameSenderInterface* sender,
      rtc::scoped_refptr<FrameTransformerInterface> frame_transformer,
      uint32_t ssrc,
      TaskQueueFactory* send_transport_queue);

  void Init();

  // Delegates the call to FrameTransformerInterface::TransformFrame.
  bool TransformFrame(int payload_type,
                      std::optional<VideoCodecType> codec_type,
                      uint32_t rtp_timestamp,
                      const EncodedImage& encoded_image,
                      RTPVideoHeader video_header,
                      TimeDelta expected_retransmission_time);

  // Implements TransformedFrameCallback. Can be called on any thread. Posts
  // the transformed frame to be sent on the `encoder_queue_`.
  void OnTransformedFrame(
      std::unique_ptr<TransformableFrameInterface> frame) override;

  void StartShortCircuiting() override;

  // Delegates the call to RTPSendVideo::SendVideo on the `encoder_queue_`.
  void SendVideo(std::unique_ptr<TransformableFrameInterface> frame) const
      RTC_RUN_ON(transformation_queue_);

  // Delegates the call to RTPSendVideo::SetVideoStructureAfterTransformation
  // under `sender_lock_`.
  void SetVideoStructureUnderLock(
      const FrameDependencyStructure* video_structure);

  // Delegates the call to
  // RTPSendVideo::SetVideoLayersAllocationAfterTransformation under
  // `sender_lock_`.
  void SetVideoLayersAllocationUnderLock(VideoLayersAllocation allocation);

  // Unregisters and releases the `frame_transformer_` reference, and resets
  // `sender_` under lock. Called from RTPSenderVideo destructor to prevent the
  // `sender_` to dangle.
  void Reset();

 protected:
  ~RTPSenderVideoFrameTransformerDelegate() override = default;

 private:
  void EnsureEncoderQueueCreated();

  mutable Mutex sender_lock_;
  RTPVideoFrameSenderInterface* sender_ RTC_GUARDED_BY(sender_lock_);
  rtc::scoped_refptr<FrameTransformerInterface> frame_transformer_;
  const uint32_t ssrc_;
  // Used when the encoded frames arrives without a current task queue. This can
  // happen if a hardware encoder was used.
  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> transformation_queue_;
  bool short_circuit_ RTC_GUARDED_BY(sender_lock_) = false;
};

// Method to support cloning a Sender frame from another frame
std::unique_ptr<TransformableVideoFrameInterface> CloneSenderVideoFrame(
    TransformableVideoFrameInterface* original);

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_SENDER_VIDEO_FRAME_TRANSFORMER_DELEGATE_H_
