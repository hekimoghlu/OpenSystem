/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_DXGI_TEXTURE_H_
#define MODULES_DESKTOP_CAPTURE_WIN_DXGI_TEXTURE_H_

#include <d3d11.h>
#include <dxgi1_2.h>

#include <memory>

#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/desktop_geometry.h"

namespace webrtc {

class DesktopRegion;

// A texture copied or mapped from a DXGI_OUTDUPL_FRAME_INFO and IDXGIResource.
class DxgiTexture {
 public:
  // Creates a DxgiTexture instance, which represents the `desktop_size` area of
  // entire screen -- usually a monitor on the system.
  DxgiTexture();

  virtual ~DxgiTexture();

  // Copies selected regions of a frame represented by frame_info and resource.
  // Returns false if anything wrong.
  bool CopyFrom(const DXGI_OUTDUPL_FRAME_INFO& frame_info,
                IDXGIResource* resource);

  const DesktopSize& desktop_size() const { return desktop_size_; }

  uint8_t* bits() const { return static_cast<uint8_t*>(rect_.pBits); }

  int pitch() const { return static_cast<int>(rect_.Pitch); }

  // Releases the resource currently holds by this instance. Returns false if
  // anything wrong, and this instance should be deprecated in this state. bits,
  // pitch and AsDesktopFrame are only valid after a success CopyFrom() call,
  // but before Release() call.
  bool Release();

  // Returns a DesktopFrame snapshot of a DxgiTexture instance. This
  // DesktopFrame is used to copy a DxgiTexture content to another DesktopFrame
  // only. And it should not outlive its DxgiTexture instance.
  const DesktopFrame& AsDesktopFrame();

 protected:
  DXGI_MAPPED_RECT* rect();

  virtual bool CopyFromTexture(const DXGI_OUTDUPL_FRAME_INFO& frame_info,
                               ID3D11Texture2D* texture) = 0;

  virtual bool DoRelease() = 0;

 private:
  DXGI_MAPPED_RECT rect_ = {0};
  DesktopSize desktop_size_;
  std::unique_ptr<DesktopFrame> frame_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_DXGI_TEXTURE_H_
