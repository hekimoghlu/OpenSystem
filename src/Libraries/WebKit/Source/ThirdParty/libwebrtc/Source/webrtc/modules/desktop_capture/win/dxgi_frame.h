/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_DXGI_FRAME_H_
#define MODULES_DESKTOP_CAPTURE_WIN_DXGI_FRAME_H_

#include <memory>
#include <vector>

#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/resolution_tracker.h"
#include "modules/desktop_capture/shared_desktop_frame.h"
#include "modules/desktop_capture/shared_memory.h"
#include "modules/desktop_capture/win/dxgi_context.h"

namespace webrtc {

class DxgiDuplicatorController;

// A pair of a SharedDesktopFrame and a DxgiDuplicatorController::Context for
// the client of DxgiDuplicatorController.
class DxgiFrame final {
 public:
  using Context = DxgiFrameContext;

  // DxgiFrame does not take ownership of `factory`, consumers should ensure it
  // outlives this instance. nullptr is acceptable.
  explicit DxgiFrame(SharedMemoryFactory* factory);
  ~DxgiFrame();

  // Should not be called if Prepare() is not executed or returns false.
  SharedDesktopFrame* frame() const;

 private:
  // Allows DxgiDuplicatorController to access Prepare() and context() function
  // as well as Context class.
  friend class DxgiDuplicatorController;

  // Prepares current instance with desktop size and source id.
  bool Prepare(DesktopSize size, DesktopCapturer::SourceId source_id);

  // Should not be called if Prepare() is not executed or returns false.
  Context* context();

  SharedMemoryFactory* const factory_;
  ResolutionTracker resolution_tracker_;
  DesktopCapturer::SourceId source_id_ = kFullDesktopScreenId;
  std::unique_ptr<SharedDesktopFrame> frame_;
  Context context_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_DXGI_FRAME_H_
