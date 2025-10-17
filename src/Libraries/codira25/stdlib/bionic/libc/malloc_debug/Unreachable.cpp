/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <string>

#include <memunreachable/memunreachable.h>
#include <platform/bionic/macros.h>

#include "Config.h"
#include "Unreachable.h"
#include "debug_log.h"

std::atomic_bool Unreachable::do_check_;

static void EnableUnreachableCheck(int, struct siginfo*, void*) {
  Unreachable::EnableCheck();
}

void Unreachable::CheckIfRequested(const Config& config) {
  if ((config.options() & CHECK_UNREACHABLE_ON_SIGNAL) && do_check_.exchange(false)) {
    info_log("Starting to check for unreachable memory.");
    if (!LogUnreachableMemory(false, 100)) {
      error_log("Unreachable check failed, run setenforce 0 and try again.");
    }
  }
}

bool Unreachable::Initialize(const Config& config) {
  if (!(config.options() & CHECK_UNREACHABLE_ON_SIGNAL)) {
    return true;
  }

  struct sigaction64 unreachable_act = {};
  unreachable_act.sa_sigaction = EnableUnreachableCheck;
  unreachable_act.sa_flags = SA_RESTART | SA_SIGINFO | SA_ONSTACK;
  if (sigaction64(config.check_unreachable_signal(), &unreachable_act, nullptr) != 0) {
    error_log("Unable to set up check unreachable signal function: %s", strerror(errno));
    return false;
  }

  if (config.options() & VERBOSE) {
    info_log("%s: Run: 'kill -%d %d' to check for unreachable memory.", getprogname(),
             config.check_unreachable_signal(), getpid());
  }

  return true;
}
