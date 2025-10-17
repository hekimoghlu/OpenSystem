/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#ifndef MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_VIDEO_CAPTURE_DS_H_
#define MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_VIDEO_CAPTURE_DS_H_

#include "api/scoped_refptr.h"
#include "modules/video_capture/video_capture_impl.h"
#include "modules/video_capture/windows/device_info_ds.h"

#define CAPTURE_FILTER_NAME L"VideoCaptureFilter"
#define SINK_FILTER_NAME L"SinkFilter"

namespace webrtc {
namespace videocapturemodule {
// Forward declaraion
class CaptureSinkFilter;

class VideoCaptureDS : public VideoCaptureImpl {
 public:
  VideoCaptureDS();

  virtual int32_t Init(const char* deviceUniqueIdUTF8);

  /*************************************************************************
   *
   *   Start/Stop
   *
   *************************************************************************/
  int32_t StartCapture(const VideoCaptureCapability& capability) override;
  int32_t StopCapture() override;

  /**************************************************************************
   *
   *   Properties of the set device
   *
   **************************************************************************/

  bool CaptureStarted() override;
  int32_t CaptureSettings(VideoCaptureCapability& settings) override;

 protected:
  ~VideoCaptureDS() override;

  // Help functions

  int32_t SetCameraOutput(const VideoCaptureCapability& requestedCapability);
  int32_t DisconnectGraph();
  HRESULT ConnectDVCamera();

  DeviceInfoDS _dsInfo RTC_GUARDED_BY(api_checker_);

  IBaseFilter* _captureFilter RTC_GUARDED_BY(api_checker_);
  IGraphBuilder* _graphBuilder RTC_GUARDED_BY(api_checker_);
  IMediaControl* _mediaControl RTC_GUARDED_BY(api_checker_);
  rtc::scoped_refptr<CaptureSinkFilter> sink_filter_
      RTC_GUARDED_BY(api_checker_);
  IPin* _inputSendPin RTC_GUARDED_BY(api_checker_);
  IPin* _outputCapturePin RTC_GUARDED_BY(api_checker_);

  // Microsoft DV interface (external DV cameras)
  IBaseFilter* _dvFilter RTC_GUARDED_BY(api_checker_);
  IPin* _inputDvPin RTC_GUARDED_BY(api_checker_);
  IPin* _outputDvPin RTC_GUARDED_BY(api_checker_);
};
}  // namespace videocapturemodule
}  // namespace webrtc
#endif  // MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_VIDEO_CAPTURE_DS_H_
