/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#include <stdlib.h>
#include <string.h>

#include "vpx_config.h"
#include "vpx_ports/arm.h"

#if defined(_WIN32)
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef WIN32_EXTRA_LEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#endif

#ifdef WINAPI_FAMILY
#include <winapifamily.h>
#if !WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#define getenv(x) NULL
#endif
#endif

#if defined(__ANDROID__) && (__ANDROID_API__ < 18)
#define VPX_USE_ANDROID_CPU_FEATURES 1
// Use getauxval() when targeting (64-bit) Android with API level >= 18.
// getauxval() is supported since Android API level 18 (Android 4.3.)
// First Android version with 64-bit support was Android 5.x (API level 21).
#include <cpu-features.h>
#endif

static INLINE int arm_cpu_env_flags(int *flags) {
  const char *env = getenv("VPX_SIMD_CAPS");
  if (env && *env) {
    *flags = (int)strtol(env, NULL, 0);
    return 1;
  }
  return 0;
}

static INLINE int arm_cpu_env_mask(void) {
  const char *env = getenv("VPX_SIMD_CAPS_MASK");
  return env && *env ? (int)strtol(env, NULL, 0) : ~0;
}
