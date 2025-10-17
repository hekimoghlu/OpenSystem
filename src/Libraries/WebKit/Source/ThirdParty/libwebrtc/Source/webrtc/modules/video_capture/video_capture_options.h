/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
#ifndef MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_OPTIONS_H_
#define MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_OPTIONS_H_

#include "api/scoped_refptr.h"
#include "rtc_base/system/rtc_export.h"

#if defined(WEBRTC_USE_PIPEWIRE)
#include "modules/portal/pipewire_utils.h"
#endif

namespace webrtc {

#if defined(WEBRTC_USE_PIPEWIRE)
namespace videocapturemodule {
class PipeWireSession;
}
#endif

// An object that stores initialization parameters for video capturers
class RTC_EXPORT VideoCaptureOptions {
 public:
  VideoCaptureOptions();
  VideoCaptureOptions(const VideoCaptureOptions& options);
  VideoCaptureOptions(VideoCaptureOptions&& options);
  ~VideoCaptureOptions();

  VideoCaptureOptions& operator=(const VideoCaptureOptions& options);
  VideoCaptureOptions& operator=(VideoCaptureOptions&& options);

  enum class Status {
    SUCCESS,
    UNINITIALIZED,
    UNAVAILABLE,
    DENIED,
    ERROR,
    MAX_VALUE = ERROR
  };

  class Callback {
   public:
    virtual void OnInitialized(Status status) = 0;

   protected:
    virtual ~Callback() = default;
  };

  void Init(Callback* callback);

#if defined(WEBRTC_LINUX)
  bool allow_v4l2() const { return allow_v4l2_; }
  void set_allow_v4l2(bool allow) { allow_v4l2_ = allow; }
#endif

#if defined(WEBRTC_USE_PIPEWIRE)
  bool allow_pipewire() const { return allow_pipewire_; }
  void set_allow_pipewire(bool allow) { allow_pipewire_ = allow; }
  void set_pipewire_fd(int fd) { pipewire_fd_ = fd; }
  rtc::scoped_refptr<videocapturemodule::PipeWireSession> pipewire_session();
#endif

 private:
#if defined(WEBRTC_LINUX)
  bool allow_v4l2_ = false;
#endif
#if defined(WEBRTC_USE_PIPEWIRE)
  bool allow_pipewire_ = false;
  int pipewire_fd_ = kInvalidPipeWireFd;
  rtc::scoped_refptr<videocapturemodule::PipeWireSession> pipewire_session_;
#endif
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_OPTIONS_H_
