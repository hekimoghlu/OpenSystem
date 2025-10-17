/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include "modules/video_capture/device_info_impl.h"

#include <stdlib.h>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "rtc_base/logging.h"

#ifndef abs
#define abs(a) (a >= 0 ? a : -a)
#endif

namespace webrtc {
namespace videocapturemodule {

DeviceInfoImpl::DeviceInfoImpl()
    : _lastUsedDeviceName(NULL), _lastUsedDeviceNameLength(0) {}

DeviceInfoImpl::~DeviceInfoImpl(void) {
  MutexLock lock(&_apiLock);
  free(_lastUsedDeviceName);
}

int32_t DeviceInfoImpl::NumberOfCapabilities(const char* deviceUniqueIdUTF8) {
  if (!deviceUniqueIdUTF8)
    return -1;

  MutexLock lock(&_apiLock);

  // Is it the same device that is asked for again.
  if (absl::EqualsIgnoreCase(
          deviceUniqueIdUTF8,
          absl::string_view(_lastUsedDeviceName, _lastUsedDeviceNameLength))) {
    return static_cast<int32_t>(_captureCapabilities.size());
  }

  int32_t ret = CreateCapabilityMap(deviceUniqueIdUTF8);
  return ret;
}

int32_t DeviceInfoImpl::GetCapability(const char* deviceUniqueIdUTF8,
                                      const uint32_t deviceCapabilityNumber,
                                      VideoCaptureCapability& capability) {
  RTC_DCHECK(deviceUniqueIdUTF8);

  MutexLock lock(&_apiLock);

  if (!absl::EqualsIgnoreCase(
          deviceUniqueIdUTF8,
          absl::string_view(_lastUsedDeviceName, _lastUsedDeviceNameLength))) {
    if (-1 == CreateCapabilityMap(deviceUniqueIdUTF8)) {
      return -1;
    }
  }

  // Make sure the number is valid
  if (deviceCapabilityNumber >= (unsigned int)_captureCapabilities.size()) {
    RTC_LOG(LS_ERROR) << "Invalid deviceCapabilityNumber "
                      << deviceCapabilityNumber << ">= number of capabilities ("
                      << _captureCapabilities.size() << ").";
    return -1;
  }

  capability = _captureCapabilities[deviceCapabilityNumber];
  return 0;
}

int32_t DeviceInfoImpl::GetBestMatchedCapability(
    const char* deviceUniqueIdUTF8,
    const VideoCaptureCapability& requested,
    VideoCaptureCapability& resulting) {
  if (!deviceUniqueIdUTF8)
    return -1;

  MutexLock lock(&_apiLock);
  if (!absl::EqualsIgnoreCase(
          deviceUniqueIdUTF8,
          absl::string_view(_lastUsedDeviceName, _lastUsedDeviceNameLength))) {
    if (-1 == CreateCapabilityMap(deviceUniqueIdUTF8)) {
      return -1;
    }
  }

  int32_t bestformatIndex = -1;
  int32_t bestWidth = 0;
  int32_t bestHeight = 0;
  int32_t bestFrameRate = 0;
  VideoType bestVideoType = VideoType::kUnknown;

  const int32_t numberOfCapabilies =
      static_cast<int32_t>(_captureCapabilities.size());

  for (int32_t tmp = 0; tmp < numberOfCapabilies;
       ++tmp)  // Loop through all capabilities
  {
    VideoCaptureCapability& capability = _captureCapabilities[tmp];

    const int32_t diffWidth = capability.width - requested.width;
    const int32_t diffHeight = capability.height - requested.height;
    const int32_t diffFrameRate = capability.maxFPS - requested.maxFPS;

    const int32_t currentbestDiffWith = bestWidth - requested.width;
    const int32_t currentbestDiffHeight = bestHeight - requested.height;
    const int32_t currentbestDiffFrameRate = bestFrameRate - requested.maxFPS;

    if ((diffHeight >= 0 &&
         diffHeight <= abs(currentbestDiffHeight))  // Height better or equalt
                                                    // that previouse.
        || (currentbestDiffHeight < 0 && diffHeight >= currentbestDiffHeight)) {
      if (diffHeight ==
          currentbestDiffHeight)  // Found best height. Care about the width)
      {
        if ((diffWidth >= 0 &&
             diffWidth <= abs(currentbestDiffWith))  // Width better or equal
            || (currentbestDiffWith < 0 && diffWidth >= currentbestDiffWith)) {
          if (diffWidth == currentbestDiffWith &&
              diffHeight == currentbestDiffHeight)  // Same size as previously
          {
            // Also check the best frame rate if the diff is the same as
            // previouse
            if (((diffFrameRate >= 0 &&
                  diffFrameRate <=
                      currentbestDiffFrameRate)  // Frame rate to high but
                                                 // better match than previouse
                                                 // and we have not selected IUV
                 || (currentbestDiffFrameRate < 0 &&
                     diffFrameRate >=
                         currentbestDiffFrameRate))  // Current frame rate is
                                                     // lower than requested.
                                                     // This is better.
            ) {
              if ((currentbestDiffFrameRate ==
                   diffFrameRate)  // Same frame rate as previous  or frame rate
                                   // allready good enough
                  || (currentbestDiffFrameRate >= 0)) {
                if (bestVideoType != requested.videoType &&
                    requested.videoType != VideoType::kUnknown &&
                    (capability.videoType == requested.videoType ||
                     capability.videoType == VideoType::kI420 ||
                     capability.videoType == VideoType::kYUY2 ||
                     capability.videoType == VideoType::kYV12 ||
                     capability.videoType == VideoType::kNV12)) {
                  bestVideoType = capability.videoType;
                  bestformatIndex = tmp;
                }
                // If width height and frame rate is full filled we can use the
                // camera for encoding if it is supported.
                if (capability.height == requested.height &&
                    capability.width == requested.width &&
                    capability.maxFPS >= requested.maxFPS) {
                  bestformatIndex = tmp;
                }
              } else  // Better frame rate
              {
                bestWidth = capability.width;
                bestHeight = capability.height;
                bestFrameRate = capability.maxFPS;
                bestVideoType = capability.videoType;
                bestformatIndex = tmp;
              }
            }
          } else  // Better width than previously
          {
            bestWidth = capability.width;
            bestHeight = capability.height;
            bestFrameRate = capability.maxFPS;
            bestVideoType = capability.videoType;
            bestformatIndex = tmp;
          }
        }     // else width no good
      } else  // Better height
      {
        bestWidth = capability.width;
        bestHeight = capability.height;
        bestFrameRate = capability.maxFPS;
        bestVideoType = capability.videoType;
        bestformatIndex = tmp;
      }
    }  // else height not good
  }    // end for

  RTC_LOG(LS_VERBOSE) << "Best camera format: " << bestWidth << "x"
                      << bestHeight << "@" << bestFrameRate
                      << "fps, color format: "
                      << static_cast<int>(bestVideoType);

  // Copy the capability
  if (bestformatIndex < 0)
    return -1;
  resulting = _captureCapabilities[bestformatIndex];
  return bestformatIndex;
}

// Default implementation. This should be overridden by Mobile implementations.
int32_t DeviceInfoImpl::GetOrientation(const char* deviceUniqueIdUTF8,
                                       VideoRotation& orientation) {
  orientation = kVideoRotation_0;
  return -1;
}
}  // namespace videocapturemodule
}  // namespace webrtc
