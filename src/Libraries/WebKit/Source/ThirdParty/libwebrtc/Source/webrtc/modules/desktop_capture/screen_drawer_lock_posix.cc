/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#include "modules/desktop_capture/screen_drawer_lock_posix.h"

#include <fcntl.h>
#include <sys/stat.h>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

namespace {

// A uuid as the name of semaphore.
static constexpr char kSemaphoreName[] = "GSDL54fe5552804711e6a7253f429a";

}  // namespace

ScreenDrawerLockPosix::ScreenDrawerLockPosix()
    : ScreenDrawerLockPosix(kSemaphoreName) {}

ScreenDrawerLockPosix::ScreenDrawerLockPosix(const char* name) {
  semaphore_ = sem_open(name, O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 1);
  if (semaphore_ == SEM_FAILED) {
    RTC_LOG_ERRNO(LS_ERROR) << "Failed to create named semaphore with " << name;
    RTC_DCHECK_NOTREACHED();
  }

  sem_wait(semaphore_);
}

ScreenDrawerLockPosix::~ScreenDrawerLockPosix() {
  if (semaphore_ == SEM_FAILED) {
    return;
  }

  sem_post(semaphore_);
  sem_close(semaphore_);
  // sem_unlink a named semaphore won't wait until other clients to release the
  // sem_t. So if a new process starts, it will sem_open a different kernel
  // object with the same name and eventually breaks the cross-process lock.
}

// static
void ScreenDrawerLockPosix::Unlink(absl::string_view name) {
  sem_unlink(std::string(name).c_str());
}

}  // namespace webrtc
