/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include "video/adaptation/video_stream_encoder_resource.h"

#include <algorithm>
#include <utility>

namespace webrtc {

VideoStreamEncoderResource::VideoStreamEncoderResource(std::string name)
    : lock_(),
      name_(std::move(name)),
      encoder_queue_(nullptr),
      listener_(nullptr) {}

VideoStreamEncoderResource::~VideoStreamEncoderResource() {
  RTC_DCHECK(!listener_)
      << "There is a listener depending on a VideoStreamEncoderResource being "
      << "destroyed.";
}

void VideoStreamEncoderResource::RegisterEncoderTaskQueue(
    TaskQueueBase* encoder_queue) {
  RTC_DCHECK(!encoder_queue_);
  RTC_DCHECK(encoder_queue);
  encoder_queue_ = encoder_queue;
}

void VideoStreamEncoderResource::SetResourceListener(
    ResourceListener* listener) {
  // If you want to change listener you need to unregister the old listener by
  // setting it to null first.
  MutexLock crit(&lock_);
  RTC_DCHECK(!listener_ || !listener) << "A listener is already set";
  listener_ = listener;
}

std::string VideoStreamEncoderResource::Name() const {
  return name_;
}

void VideoStreamEncoderResource::OnResourceUsageStateMeasured(
    ResourceUsageState usage_state) {
  MutexLock crit(&lock_);
  if (listener_) {
    listener_->OnResourceUsageStateMeasured(rtc::scoped_refptr<Resource>(this),
                                            usage_state);
  }
}

TaskQueueBase* VideoStreamEncoderResource::encoder_queue() const {
  return encoder_queue_;
}

}  // namespace webrtc
