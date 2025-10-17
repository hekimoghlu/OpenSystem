/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#include "api/frame_transformer_factory.h"

#include <memory>

#include "api/frame_transformer_interface.h"
#include "audio/channel_receive_frame_transformer_delegate.h"
#include "audio/channel_send_frame_transformer_delegate.h"
#include "modules/rtp_rtcp/source/rtp_sender_video_frame_transformer_delegate.h"
#include "rtc_base/checks.h"

namespace webrtc {

std::unique_ptr<TransformableVideoFrameInterface> CreateVideoSenderFrame() {
  RTC_CHECK_NOTREACHED();
  return nullptr;
}

std::unique_ptr<TransformableVideoFrameInterface> CreateVideoReceiverFrame() {
  RTC_CHECK_NOTREACHED();
  return nullptr;
}

std::unique_ptr<TransformableAudioFrameInterface> CloneAudioFrame(
    TransformableAudioFrameInterface* original) {
  if (original->GetDirection() ==
      TransformableAudioFrameInterface::Direction::kReceiver)
    return CloneReceiverAudioFrame(original);
  return CloneSenderAudioFrame(original);
}

std::unique_ptr<TransformableVideoFrameInterface> CloneVideoFrame(
    TransformableVideoFrameInterface* original) {
  // At the moment, only making sender frames from receiver frames is
  // supported.
  return CloneSenderVideoFrame(original);
}

}  // namespace webrtc
