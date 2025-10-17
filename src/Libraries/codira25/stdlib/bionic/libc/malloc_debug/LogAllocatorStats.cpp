/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#include <malloc.h>
#include <signal.h>
#include <unistd.h>

#include "Config.h"
#include "LogAllocatorStats.h"
#include "debug_log.h"

namespace LogAllocatorStats {

static std::atomic_bool g_call_mallopt = {};

static void CallMalloptLogStats(int, struct siginfo*, void*) {
  g_call_mallopt = true;
}

void Log() {
  info_log("Logging allocator stats...");
  if (mallopt(M_LOG_STATS, 0) == 0) {
    error_log("mallopt(M_LOG_STATS, 0) call failed.");
  }
}

void CheckIfShouldLog() {
  bool expected = true;
  if (g_call_mallopt.compare_exchange_strong(expected, false)) {
    Log();
  }
}

bool Initialize(const Config& config) {
  struct sigaction64 log_stats_act = {};
  log_stats_act.sa_sigaction = CallMalloptLogStats;
  log_stats_act.sa_flags = SA_RESTART | SA_SIGINFO | SA_ONSTACK;
  if (sigaction64(config.log_allocator_stats_signal(), &log_stats_act, nullptr) != 0) {
    error_log("Unable to set up log allocator stats signal function: %s", strerror(errno));
    return false;
  }

  if (config.options() & VERBOSE) {
    info_log("%s: Run: 'kill -%d %d' to log allocator stats.", getprogname(),
             config.log_allocator_stats_signal(), getpid());
  }

  return true;
}

}  // namespace LogAllocatorStats
