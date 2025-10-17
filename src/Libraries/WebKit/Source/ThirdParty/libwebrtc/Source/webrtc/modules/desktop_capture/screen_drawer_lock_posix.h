/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_SCREEN_DRAWER_LOCK_POSIX_H_
#define MODULES_DESKTOP_CAPTURE_SCREEN_DRAWER_LOCK_POSIX_H_

#include <semaphore.h>

#include "absl/strings/string_view.h"
#include "modules/desktop_capture/screen_drawer.h"

namespace webrtc {

class ScreenDrawerLockPosix final : public ScreenDrawerLock {
 public:
  ScreenDrawerLockPosix();
  // Provides a name other than the default one for test only.
  explicit ScreenDrawerLockPosix(const char* name);
  ~ScreenDrawerLockPosix() override;

  // Unlinks the named semaphore actively. This will remove the sem_t object in
  // the system and allow others to create a different sem_t object with the
  // same/ name.
  static void Unlink(absl::string_view name);

 private:
  sem_t* semaphore_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_SCREEN_DRAWER_LOCK_POSIX_H_
