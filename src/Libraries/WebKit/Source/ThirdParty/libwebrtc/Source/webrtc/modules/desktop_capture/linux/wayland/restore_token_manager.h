/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_RESTORE_TOKEN_MANAGER_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_RESTORE_TOKEN_MANAGER_H_

#include <mutex>
#include <string>
#include <unordered_map>

#include "modules/desktop_capture/desktop_capturer.h"

namespace webrtc {

class RestoreTokenManager {
 public:
  RestoreTokenManager(const RestoreTokenManager& manager) = delete;
  RestoreTokenManager& operator=(const RestoreTokenManager& manager) = delete;

  static RestoreTokenManager& GetInstance();

  void AddToken(DesktopCapturer::SourceId id, const std::string& token);
  std::string GetToken(DesktopCapturer::SourceId id);

  // Returns a source ID which does not have any token associated with it yet.
  DesktopCapturer::SourceId GetUnusedId();

 private:
  RestoreTokenManager() = default;
  ~RestoreTokenManager() = default;

  DesktopCapturer::SourceId last_source_id_ = 0;

  std::unordered_map<DesktopCapturer::SourceId, std::string> restore_tokens_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_RESTORE_TOKEN_MANAGER_H_
