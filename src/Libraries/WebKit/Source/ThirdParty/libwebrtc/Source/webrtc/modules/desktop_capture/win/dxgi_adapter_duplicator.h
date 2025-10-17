/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_DXGI_ADAPTER_DUPLICATOR_H_
#define MODULES_DESKTOP_CAPTURE_WIN_DXGI_ADAPTER_DUPLICATOR_H_

#include <wrl/client.h>

#include <vector>

#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/shared_desktop_frame.h"
#include "modules/desktop_capture/win/d3d_device.h"
#include "modules/desktop_capture/win/dxgi_context.h"
#include "modules/desktop_capture/win/dxgi_output_duplicator.h"

namespace webrtc {

// A container of DxgiOutputDuplicators to duplicate monitors attached to a
// single video card.
class DxgiAdapterDuplicator {
 public:
  using Context = DxgiAdapterContext;

  // Creates an instance of DxgiAdapterDuplicator from a D3dDevice. Only
  // DxgiDuplicatorController can create an instance.
  explicit DxgiAdapterDuplicator(const D3dDevice& device);

  // Move constructor, to make it possible to store instances of
  // DxgiAdapterDuplicator in std::vector<>.
  DxgiAdapterDuplicator(DxgiAdapterDuplicator&& other);

  ~DxgiAdapterDuplicator();

  // Initializes the DxgiAdapterDuplicator from a D3dDevice.
  bool Initialize();

  // Sequentially calls Duplicate function of all the DxgiOutputDuplicator
  // instances owned by this instance, and writes into `target`.
  bool Duplicate(Context* context, SharedDesktopFrame* target);

  // Captures one monitor and writes into `target`. `monitor_id` should be
  // between [0, screen_count()).
  bool DuplicateMonitor(Context* context,
                        int monitor_id,
                        SharedDesktopFrame* target);

  // Returns desktop rect covered by this DxgiAdapterDuplicator.
  DesktopRect desktop_rect() const { return desktop_rect_; }

  // Returns the size of one screen owned by this DxgiAdapterDuplicator. `id`
  // should be between [0, screen_count()).
  DesktopRect ScreenRect(int id) const;

  // Returns the device name of one screen owned by this DxgiAdapterDuplicator
  // in utf8 encoding. `id` should be between [0, screen_count()).
  const std::string& GetDeviceName(int id) const;

  // Returns the count of screens owned by this DxgiAdapterDuplicator. These
  // screens can be retrieved by an interger in the range of
  // [0, screen_count()).
  int screen_count() const;

  void Setup(Context* context);

  void Unregister(const Context* const context);

  // The minimum num_frames_captured() returned by `duplicators_`.
  int64_t GetNumFramesCaptured() const;

  // Moves `desktop_rect_` and all underlying `duplicators_`. See
  // DxgiDuplicatorController::TranslateRect().
  void TranslateRect(const DesktopVector& position);

 private:
  bool DoInitialize();

  const D3dDevice device_;
  std::vector<DxgiOutputDuplicator> duplicators_;
  DesktopRect desktop_rect_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_DXGI_ADAPTER_DUPLICATOR_H_
