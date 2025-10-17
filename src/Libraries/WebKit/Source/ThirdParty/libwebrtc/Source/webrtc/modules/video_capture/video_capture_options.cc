/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include "modules/video_capture/video_capture_options.h"

#if defined(WEBRTC_USE_PIPEWIRE)
#include "modules/video_capture/linux/pipewire_session.h"
#endif

namespace webrtc {

VideoCaptureOptions::VideoCaptureOptions() {}
VideoCaptureOptions::VideoCaptureOptions(const VideoCaptureOptions& options) =
    default;
VideoCaptureOptions::VideoCaptureOptions(VideoCaptureOptions&& options) =
    default;
VideoCaptureOptions::~VideoCaptureOptions() {}

VideoCaptureOptions& VideoCaptureOptions::operator=(
    const VideoCaptureOptions& options) = default;
VideoCaptureOptions& VideoCaptureOptions::operator=(
    VideoCaptureOptions&& options) = default;

void VideoCaptureOptions::Init(Callback* callback) {
#if defined(WEBRTC_USE_PIPEWIRE)
  if (allow_pipewire_) {
    pipewire_session_ =
        rtc::make_ref_counted<videocapturemodule::PipeWireSession>();
    pipewire_session_->Init(callback, pipewire_fd_);
    return;
  }
#endif
#if defined(WEBRTC_LINUX)
  if (!allow_v4l2_)
    callback->OnInitialized(Status::UNAVAILABLE);
  else
#endif
    callback->OnInitialized(Status::SUCCESS);
}

#if defined(WEBRTC_USE_PIPEWIRE)
rtc::scoped_refptr<videocapturemodule::PipeWireSession>
VideoCaptureOptions::pipewire_session() {
  return pipewire_session_;
}
#endif

}  // namespace webrtc
