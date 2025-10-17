/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#include "modules/desktop_capture/win/dxgi_texture_mapping.h"

#include <comdef.h>
#include <dxgi.h>
#include <dxgi1_2.h>

#include "modules/desktop_capture/win/desktop_capture_utils.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

DxgiTextureMapping::DxgiTextureMapping(IDXGIOutputDuplication* duplication)
    : duplication_(duplication) {
  RTC_DCHECK(duplication_);
}

DxgiTextureMapping::~DxgiTextureMapping() = default;

bool DxgiTextureMapping::CopyFromTexture(
    const DXGI_OUTDUPL_FRAME_INFO& frame_info,
    ID3D11Texture2D* texture) {
  RTC_DCHECK_GT(frame_info.AccumulatedFrames, 0);
  RTC_DCHECK(texture);
  *rect() = {0};
  _com_error error = duplication_->MapDesktopSurface(rect());
  if (error.Error() != S_OK) {
    *rect() = {0};
    RTC_LOG(LS_ERROR)
        << "Failed to map the IDXGIOutputDuplication to a bitmap: "
        << desktop_capture::utils::ComErrorToString(error);
    return false;
  }

  return true;
}

bool DxgiTextureMapping::DoRelease() {
  _com_error error = duplication_->UnMapDesktopSurface();
  if (error.Error() != S_OK) {
    RTC_LOG(LS_ERROR) << "Failed to unmap the IDXGIOutputDuplication: "
                      << desktop_capture::utils::ComErrorToString(error);
    return false;
  }
  return true;
}

}  // namespace webrtc
