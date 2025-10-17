/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_DXGI_TEXTURE_MAPPING_H_
#define MODULES_DESKTOP_CAPTURE_WIN_DXGI_TEXTURE_MAPPING_H_

#include <d3d11.h>
#include <dxgi1_2.h>

#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/desktop_region.h"
#include "modules/desktop_capture/win/dxgi_texture.h"

namespace webrtc {

// A DxgiTexture which directly maps bitmap from IDXGIResource. This class is
// used when DXGI_OUTDUPL_DESC.DesktopImageInSystemMemory is true. (This usually
// means the video card shares main memory with CPU, instead of having its own
// individual memory.)
class DxgiTextureMapping : public DxgiTexture {
 public:
  // Creates a DxgiTextureMapping instance. Caller must maintain the lifetime
  // of input `duplication` to make sure it outlives this instance.
  explicit DxgiTextureMapping(IDXGIOutputDuplication* duplication);

  ~DxgiTextureMapping() override;

 protected:
  bool CopyFromTexture(const DXGI_OUTDUPL_FRAME_INFO& frame_info,
                       ID3D11Texture2D* texture) override;

  bool DoRelease() override;

 private:
  IDXGIOutputDuplication* const duplication_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_DXGI_TEXTURE_MAPPING_H_
