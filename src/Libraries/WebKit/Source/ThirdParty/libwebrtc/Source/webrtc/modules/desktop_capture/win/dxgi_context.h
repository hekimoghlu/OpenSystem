/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_DXGI_CONTEXT_H_
#define MODULES_DESKTOP_CAPTURE_WIN_DXGI_CONTEXT_H_

#include <vector>

#include "modules/desktop_capture/desktop_region.h"

namespace webrtc {

// A DxgiOutputContext stores the status of a single DxgiFrame of
// DxgiOutputDuplicator.
struct DxgiOutputContext final {
  // The updated region DxgiOutputDuplicator::DetectUpdatedRegion() output
  // during last Duplicate() function call. It's always relative to the (0, 0).
  DesktopRegion updated_region;
};

// A DxgiAdapterContext stores the status of a single DxgiFrame of
// DxgiAdapterDuplicator.
struct DxgiAdapterContext final {
  DxgiAdapterContext();
  DxgiAdapterContext(const DxgiAdapterContext& other);
  ~DxgiAdapterContext();

  // Child DxgiOutputContext belongs to this AdapterContext.
  std::vector<DxgiOutputContext> contexts;
};

// A DxgiFrameContext stores the status of a single DxgiFrame of
// DxgiDuplicatorController.
struct DxgiFrameContext final {
 public:
  DxgiFrameContext();
  // Unregister this Context instance from DxgiDuplicatorController during
  // destructing.
  ~DxgiFrameContext();

  // Reset current Context, so it will be reinitialized next time.
  void Reset();

  // A Context will have an exactly same `controller_id` as
  // DxgiDuplicatorController, to ensure it has been correctly setted up after
  // each DxgiDuplicatorController::Initialize().
  int controller_id = 0;

  // Child DxgiAdapterContext belongs to this DxgiFrameContext.
  std::vector<DxgiAdapterContext> contexts;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_DXGI_CONTEXT_H_
