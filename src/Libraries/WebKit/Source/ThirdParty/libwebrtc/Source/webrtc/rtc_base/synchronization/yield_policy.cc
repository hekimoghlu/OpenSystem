/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
#include "rtc_base/synchronization/yield_policy.h"

#include "absl/base/attributes.h"
#include "absl/base/config.h"
#include "rtc_base/checks.h"
#if !defined(ABSL_HAVE_THREAD_LOCAL) && defined(WEBRTC_POSIX)
#include <pthread.h>
#endif

namespace rtc {
namespace {

#if defined(ABSL_HAVE_THREAD_LOCAL)

ABSL_CONST_INIT thread_local YieldInterface* current_yield_policy = nullptr;

YieldInterface* GetCurrentYieldPolicy() {
  return current_yield_policy;
}

void SetCurrentYieldPolicy(YieldInterface* ptr) {
  current_yield_policy = ptr;
}

#elif defined(WEBRTC_POSIX)

// Emscripten does not support the C++11 thread_local keyword but does support
// the pthread thread-local storage API.
// https://github.com/emscripten-core/emscripten/issues/3502

ABSL_CONST_INIT pthread_key_t g_current_yield_policy_tls = 0;

void InitializeTls() {
  RTC_CHECK_EQ(pthread_key_create(&g_current_yield_policy_tls, nullptr), 0);
}

pthread_key_t GetCurrentYieldPolicyTls() {
  static pthread_once_t init_once = PTHREAD_ONCE_INIT;
  RTC_CHECK_EQ(pthread_once(&init_once, &InitializeTls), 0);
  return g_current_yield_policy_tls;
}

YieldInterface* GetCurrentYieldPolicy() {
  return static_cast<YieldInterface*>(
      pthread_getspecific(GetCurrentYieldPolicyTls()));
}

void SetCurrentYieldPolicy(YieldInterface* ptr) {
  pthread_setspecific(GetCurrentYieldPolicyTls(), ptr);
}

#else
#error Unsupported platform
#endif

}  // namespace

ScopedYieldPolicy::ScopedYieldPolicy(YieldInterface* policy)
    : previous_(GetCurrentYieldPolicy()) {
  SetCurrentYieldPolicy(policy);
}

ScopedYieldPolicy::~ScopedYieldPolicy() {
  SetCurrentYieldPolicy(previous_);
}

void ScopedYieldPolicy::YieldExecution() {
  YieldInterface* current = GetCurrentYieldPolicy();
  if (current)
    current->YieldExecution();
}

}  // namespace rtc
