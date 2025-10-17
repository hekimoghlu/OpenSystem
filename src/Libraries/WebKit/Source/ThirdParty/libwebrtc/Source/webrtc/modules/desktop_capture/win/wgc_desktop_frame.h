/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_WGC_DESKTOP_FRAME_H_
#define MODULES_DESKTOP_CAPTURE_WIN_WGC_DESKTOP_FRAME_H_

#include <d3d11.h>
#include <wrl/client.h>

#include <memory>
#include <vector>

#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/desktop_geometry.h"

namespace webrtc {

// DesktopFrame implementation used by capturers that use the
// Windows.Graphics.Capture API.
class WgcDesktopFrame final : public DesktopFrame {
 public:
  // WgcDesktopFrame receives an rvalue reference to the `image_data` vector
  // so that it can take ownership of it (and avoid a copy).
  WgcDesktopFrame(DesktopSize size,
                  int stride,
                  std::vector<uint8_t>&& image_data);

  WgcDesktopFrame(const WgcDesktopFrame&) = delete;
  WgcDesktopFrame& operator=(const WgcDesktopFrame&) = delete;

  ~WgcDesktopFrame() override;

 private:
  std::vector<uint8_t> image_data_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_WGC_DESKTOP_FRAME_H_
