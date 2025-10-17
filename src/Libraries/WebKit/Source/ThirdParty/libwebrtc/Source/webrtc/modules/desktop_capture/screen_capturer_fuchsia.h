/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURER_FUCHSIA_H_
#define MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURER_FUCHSIA_H_

#include <fuchsia/sysmem2/cpp/fidl.h>
#include <fuchsia/ui/composition/cpp/fidl.h>
#include <lib/sys/cpp/component_context.h>

#include <memory>
#include <unordered_map>

#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_frame.h"

namespace webrtc {

class ScreenCapturerFuchsia final : public DesktopCapturer {
 public:
  ScreenCapturerFuchsia();
  ~ScreenCapturerFuchsia() override;

  // DesktopCapturer interface.
  void Start(Callback* callback) override;
  void CaptureFrame() override;
  bool GetSourceList(SourceList* screens) override;
  bool SelectSource(SourceId id) override;

 private:
  fuchsia::sysmem2::BufferCollectionConstraints GetBufferConstraints();
  void SetupBuffers();
  uint32_t GetPixelsPerRow(
      const fuchsia::sysmem2::ImageFormatConstraints& constraints);

  Callback* callback_ = nullptr;

  std::unique_ptr<sys::ComponentContext> component_context_;
  fuchsia::sysmem2::AllocatorSyncPtr sysmem_allocator_;
  fuchsia::ui::composition::AllocatorSyncPtr flatland_allocator_;
  fuchsia::ui::composition::ScreenCaptureSyncPtr screen_capture_;
  fuchsia::sysmem2::BufferCollectionSyncPtr collection_;
  fuchsia::sysmem2::BufferCollectionInfo buffer_collection_info_;
  std::unordered_map<uint32_t, uint8_t*> virtual_memory_mapped_addrs_;

  bool fatal_error_;

  // Dimensions of the screen we are capturing
  uint32_t width_;
  uint32_t height_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURER_FUCHSIA_H_
