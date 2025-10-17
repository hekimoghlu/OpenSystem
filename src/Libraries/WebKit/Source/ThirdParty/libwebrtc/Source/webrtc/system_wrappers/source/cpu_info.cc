/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
#include "system_wrappers/include/cpu_info.h"

#if defined(WEBRTC_WIN)
#include <windows.h>
#elif defined(WEBRTC_LINUX)
#include <unistd.h>
#elif defined(WEBRTC_MAC)
#include <sys/sysctl.h>
#elif defined(WEBRTC_FUCHSIA)
#include <zircon/syscalls.h>
#endif

#include "rtc_base/logging.h"

namespace internal {
static int DetectNumberOfCores() {
  int number_of_cores;

#if defined(WEBRTC_WIN)
  SYSTEM_INFO si;
  GetNativeSystemInfo(&si);
  number_of_cores = static_cast<int>(si.dwNumberOfProcessors);
#elif defined(WEBRTC_LINUX) || defined(WEBRTC_ANDROID)
  number_of_cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
  if (number_of_cores <= 0) {
    RTC_LOG(LS_ERROR) << "Failed to get number of cores";
    number_of_cores = 1;
  }
#elif defined(WEBRTC_MAC) || defined(WEBRTC_IOS)
  int name[] = {CTL_HW, HW_AVAILCPU};
  size_t size = sizeof(number_of_cores);
  if (0 != sysctl(name, 2, &number_of_cores, &size, NULL, 0)) {
    RTC_LOG(LS_ERROR) << "Failed to get number of cores";
    number_of_cores = 1;
  }
#elif defined(WEBRTC_FUCHSIA)
  number_of_cores = zx_system_get_num_cpus();
#else
  RTC_LOG(LS_ERROR) << "No function to get number of cores";
  number_of_cores = 1;
#endif

  RTC_LOG(LS_INFO) << "Available number of cores: " << number_of_cores;

  RTC_CHECK_GT(number_of_cores, 0);
  return number_of_cores;
}
}  // namespace internal

namespace webrtc {

uint32_t CpuInfo::DetectNumberOfCores() {
  // Statically cache the number of system cores available since if the process
  // is running in a sandbox, we may only be able to read the value once (before
  // the sandbox is initialized) and not thereafter.
  // For more information see crbug.com/176522.
  static const uint32_t logical_cpus =
      static_cast<uint32_t>(internal::DetectNumberOfCores());
  return logical_cpus;
}

}  // namespace webrtc
