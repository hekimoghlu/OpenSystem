/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
// Feature detection code for Armv7-A / AArch32.

#include "./vpx_config.h"
#include "arm_cpudetect.h"

#if !CONFIG_RUNTIME_CPU_DETECT

static int arm_get_cpu_caps(void) {
  // This function should actually be a no-op. There is no way to adjust any of
  // these because the RTCD tables do not exist: the functions are called
  // statically.
  int flags = 0;
#if HAVE_NEON
  flags |= HAS_NEON;
#endif  // HAVE_NEON
  return flags;
}

#elif defined(_MSC_VER)  // end !CONFIG_RUNTIME_CPU_DETECT

static int arm_get_cpu_caps(void) {
  int flags = 0;
#if HAVE_NEON || HAVE_NEON_ASM
  // MSVC has no inline __asm support for Arm, but it does let you __emit
  // instructions via their assembled hex code.
  // All of these instructions should be essentially nops.
  __try {
    // VORR q0,q0,q0
    __emit(0xF2200150);
    flags |= HAS_NEON;
  } __except (GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION) {
    // Ignore exception.
  }
#endif  // HAVE_NEON || HAVE_NEON_ASM
  return flags;
}

#elif defined(VPX_USE_ANDROID_CPU_FEATURES)

static int arm_get_cpu_caps(void) {
  int flags = 0;
#if HAVE_NEON || HAVE_NEON_ASM
  uint64_t features = android_getCpuFeatures();
  if (features & ANDROID_CPU_ARM_FEATURE_NEON) {
    flags |= HAS_NEON;
  }
#endif  // HAVE_NEON || HAVE_NEON_ASM
  return flags;
}

#elif defined(__linux__)  // end defined(VPX_USE_ANDROID_CPU_FEATURES)

#include <sys/auxv.h>

// Define hwcap values ourselves: building with an old auxv header where these
// hwcap values are not defined should not prevent features from being enabled.
#define VPX_AARCH32_HWCAP_NEON (1 << 12)

static int arm_get_cpu_caps(void) {
  int flags = 0;
  unsigned long hwcap = getauxval(AT_HWCAP);
#if HAVE_NEON || HAVE_NEON_ASM
  if (hwcap & VPX_AARCH32_HWCAP_NEON) {
    flags |= HAS_NEON;
  }
#endif  // HAVE_NEON || HAVE_NEON_ASM
  return flags;
}
#else   // end __linux__
#error \
    "Runtime CPU detection selected, but no CPU detection method available" \
"for your platform. Rerun configure with --disable-runtime-cpu-detect."
#endif

int arm_cpu_caps(void) {
  int flags = 0;
  if (arm_cpu_env_flags(&flags)) {
    return flags;
  }
  return arm_get_cpu_caps() & arm_cpu_env_mask();
}
