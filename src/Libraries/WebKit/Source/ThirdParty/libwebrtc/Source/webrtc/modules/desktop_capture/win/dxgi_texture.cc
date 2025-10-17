/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 6, 2022.
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
#include "modules/desktop_capture/win/dxgi_texture.h"

#include <comdef.h>
#include <d3d11.h>
#include <wrl/client.h>

#include "modules/desktop_capture/desktop_region.h"
#include "modules/desktop_capture/win/desktop_capture_utils.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

using Microsoft::WRL::ComPtr;

namespace webrtc {

namespace {

class DxgiDesktopFrame : public DesktopFrame {
 public:
  explicit DxgiDesktopFrame(const DxgiTexture& texture)
      : DesktopFrame(texture.desktop_size(),
                     texture.pitch(),
                     texture.bits(),
                     nullptr) {}

  ~DxgiDesktopFrame() override = default;
};

}  // namespace

DxgiTexture::DxgiTexture() = default;
DxgiTexture::~DxgiTexture() = default;

bool DxgiTexture::CopyFrom(const DXGI_OUTDUPL_FRAME_INFO& frame_info,
                           IDXGIResource* resource) {
  RTC_DCHECK_GT(frame_info.AccumulatedFrames, 0);
  RTC_DCHECK(resource);
  ComPtr<ID3D11Texture2D> texture;
  _com_error error = resource->QueryInterface(
      __uuidof(ID3D11Texture2D),
      reinterpret_cast<void**>(texture.GetAddressOf()));
  if (error.Error() != S_OK || !texture) {
    RTC_LOG(LS_ERROR) << "Failed to convert IDXGIResource to ID3D11Texture2D: "
                      << desktop_capture::utils::ComErrorToString(error);
    return false;
  }

  D3D11_TEXTURE2D_DESC desc = {0};
  texture->GetDesc(&desc);
  desktop_size_.set(desc.Width, desc.Height);

  return CopyFromTexture(frame_info, texture.Get());
}

const DesktopFrame& DxgiTexture::AsDesktopFrame() {
  if (!frame_) {
    frame_.reset(new DxgiDesktopFrame(*this));
  }
  return *frame_;
}

bool DxgiTexture::Release() {
  frame_.reset();
  return DoRelease();
}

DXGI_MAPPED_RECT* DxgiTexture::rect() {
  return &rect_;
}

}  // namespace webrtc
