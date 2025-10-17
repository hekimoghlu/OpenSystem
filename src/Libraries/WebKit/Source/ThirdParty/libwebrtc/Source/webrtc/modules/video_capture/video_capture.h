/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
#ifndef MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_H_
#define MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_H_

#include "api/video/video_rotation.h"
#include "api/video/video_sink_interface.h"
#include "modules/video_capture/raw_video_sink_interface.h"
#include "modules/video_capture/video_capture_defines.h"

namespace webrtc {

class VideoCaptureModule : public RefCountInterface {
 public:
  // Interface for receiving information about available camera devices.
  class DeviceInfo {
   public:
    virtual uint32_t NumberOfDevices() = 0;

    // Returns the available capture devices.
    // deviceNumber   - Index of capture device.
    // deviceNameUTF8 - Friendly name of the capture device.
    // deviceUniqueIdUTF8 - Unique name of the capture device if it exist.
    //                      Otherwise same as deviceNameUTF8.
    // productUniqueIdUTF8 - Unique product id if it exist.
    //                       Null terminated otherwise.
    virtual int32_t GetDeviceName(uint32_t deviceNumber,
                                  char* deviceNameUTF8,
                                  uint32_t deviceNameLength,
                                  char* deviceUniqueIdUTF8,
                                  uint32_t deviceUniqueIdUTF8Length,
                                  char* productUniqueIdUTF8 = 0,
                                  uint32_t productUniqueIdUTF8Length = 0) = 0;

    // Returns the number of capabilities this device.
    virtual int32_t NumberOfCapabilities(const char* deviceUniqueIdUTF8) = 0;

    // Gets the capabilities of the named device.
    virtual int32_t GetCapability(const char* deviceUniqueIdUTF8,
                                  uint32_t deviceCapabilityNumber,
                                  VideoCaptureCapability& capability) = 0;

    // Gets clockwise angle the captured frames should be rotated in order
    // to be displayed correctly on a normally rotated display.
    virtual int32_t GetOrientation(const char* deviceUniqueIdUTF8,
                                   VideoRotation& orientation) = 0;

    // Gets the capability that best matches the requested width, height and
    // frame rate.
    // Returns the deviceCapabilityNumber on success.
    virtual int32_t GetBestMatchedCapability(
        const char* deviceUniqueIdUTF8,
        const VideoCaptureCapability& requested,
        VideoCaptureCapability& resulting) = 0;

    // Display OS /capture device specific settings dialog
    virtual int32_t DisplayCaptureSettingsDialogBox(
        const char* deviceUniqueIdUTF8,
        const char* dialogTitleUTF8,
        void* parentWindow,
        uint32_t positionX,
        uint32_t positionY) = 0;

    virtual ~DeviceInfo() {}
  };

  //   Register capture data callback
  virtual void RegisterCaptureDataCallback(
      rtc::VideoSinkInterface<VideoFrame>* dataCallback) = 0;
  virtual void RegisterCaptureDataCallback(
      RawVideoSinkInterface* dataCallback) = 0;

  //  Remove capture data callback
  virtual void DeRegisterCaptureDataCallback() = 0;

  // Start capture device
  virtual int32_t StartCapture(const VideoCaptureCapability& capability) = 0;

  virtual int32_t StopCapture() = 0;

  // Returns the name of the device used by this module.
  virtual const char* CurrentDeviceName() const = 0;

  // Returns true if the capture device is running
  virtual bool CaptureStarted() = 0;

  // Gets the current configuration.
  virtual int32_t CaptureSettings(VideoCaptureCapability& settings) = 0;

  // Set the rotation of the captured frames.
  // If the rotation is set to the same as returned by
  // DeviceInfo::GetOrientation the captured frames are
  // displayed correctly if rendered.
  virtual int32_t SetCaptureRotation(VideoRotation rotation) = 0;

  // Tells the capture module whether to apply the pending rotation. By default,
  // the rotation is applied and the generated frame is up right. When set to
  // false, generated frames will carry the rotation information from
  // SetCaptureRotation. Return value indicates whether this operation succeeds.
  virtual bool SetApplyRotation(bool enable) = 0;

  // Return whether the rotation is applied or left pending.
  virtual bool GetApplyRotation() = 0;

 protected:
  ~VideoCaptureModule() override {}
};

}  // namespace webrtc
#endif  // MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_H_
