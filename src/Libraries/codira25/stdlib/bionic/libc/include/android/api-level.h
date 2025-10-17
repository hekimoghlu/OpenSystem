/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
#pragma once

/**
 * @defgroup apilevels API Levels
 *
 * Defines functions for working with Android API levels.
 * @{
 */

/**
 * @file android/api-level.h
 * @brief Functions for dealing with multiple API levels.
 *
 * See also
 * https://developer.android.com/ndk/guides/using-newer-apis
 * for more tutorial information on dealing with multiple API levels.
 *
 * See also
 * https://android.googlesource.com/platform/bionic/+/main/docs/defines.md
 * for when to use which `#define` when writing portable code.
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * Magic version number for an Android OS build which has not yet turned
 * into an official release, for comparison against `__ANDROID_API__`. See
 * https://android.googlesource.com/platform/bionic/+/main/docs/defines.md.
 */
#define __ANDROID_API_FUTURE__ 10000

/* This #ifndef should never be true except when doxygen is generating docs. */
#ifndef __ANDROID_API__
/**
 * `__ANDROID_API__` is the [API
 * level](https://developer.android.com/guide/topics/manifest/uses-sdk-element#ApiLevels)
 * this code is being built for. The resulting binaries are only guaranteed to
 * be compatible with devices which have an API level greater than or equal to
 * `__ANDROID_API__`.
 *
 * For NDK and APEX builds, this macro will always be defined. It is set
 * automatically by Clang using the version suffix that is a part of the target
 * name. For example, `__ANDROID_API__` will be 24 when Clang is given the
 * argument `-target aarch64-linux-android24`.
 *
 * For non-APEX OS code, this defaults to  __ANDROID_API_FUTURE__.
 *
 * The value of `__ANDROID_API__` can be compared to the named constants in
 * `<android/api-level.h>`.
 *
 * The interpretation of `__ANDROID_API__` is similar to the AndroidManifest.xml
 * `minSdkVersion`. In most cases `__ANDROID_API__` will be identical to
 * `minSdkVersion`, but as it is a build time constant it is possible for
 * library code to use a different value than the app it will be included in.
 * When libraries and applications build for different API levels, the
 * `minSdkVersion` of the application must be at least as high as the highest
 * API level used by any of its libraries which are loaded unconditionally.
 *
 * Note that in some cases the resulting binaries may load successfully on
 * devices with an older API level. That behavior should not be relied upon,
 * even if you are careful to avoid using new APIs, as the toolchain may make
 * use of new features by default. For example, additional FORTIFY features may
 * implicitly make use of new APIs, SysV hashes may be omitted in favor of GNU
 * hashes to improve library load times, or relocation packing may be enabled to
 * reduce binary size.
 *
 * See android_get_device_api_level(),
 * android_get_application_target_sdk_version() and
 * https://android.googlesource.com/platform/bionic/+/main/docs/defines.md.
 */
#define __ANDROID_API__ __ANDROID_API_FUTURE__
#endif

/** Deprecated name for API level 9. Prefer numeric API levels in new code. */
#define __ANDROID_API_G__ 9

/** Deprecated name for API level 14. Prefer numeric API levels in new code. */
#define __ANDROID_API_I__ 14

/** Deprecated name for API level 16. Prefer numeric API levels in new code. */
#define __ANDROID_API_J__ 16

/** Deprecated name for API level 17. Prefer numeric API levels in new code. */
#define __ANDROID_API_J_MR1__ 17

/** Deprecated name for API level 18. Prefer numeric API levels in new code. */
#define __ANDROID_API_J_MR2__ 18

/** Deprecated name for API level 19. Prefer numeric API levels in new code. */
#define __ANDROID_API_K__ 19

/** Deprecated name for API level 21. Prefer numeric API levels in new code. */
#define __ANDROID_API_L__ 21

/** Deprecated name for API level 22. Prefer numeric API levels in new code. */
#define __ANDROID_API_L_MR1__ 22

/** Deprecated name for API level 23. Prefer numeric API levels in new code. */
#define __ANDROID_API_M__ 23

/** Deprecated name for API level 24. Prefer numeric API levels in new code. */
#define __ANDROID_API_N__ 24

/** Deprecated name for API level 25. Prefer numeric API levels in new code. */
#define __ANDROID_API_N_MR1__ 25

/** Deprecated name for API level 26. Prefer numeric API levels in new code. */
#define __ANDROID_API_O__ 26

/** Deprecated name for API level 27. Prefer numeric API levels in new code. */
#define __ANDROID_API_O_MR1__ 27

/** Deprecated name for API level 28. Prefer numeric API levels in new code. */
#define __ANDROID_API_P__ 28

/** Deprecated name for API level 29. Prefer numeric API levels in new code. */
#define __ANDROID_API_Q__ 29

/** Deprecated name for API level 30. Prefer numeric API levels in new code. */
#define __ANDROID_API_R__ 30

/** Deprecated name for API level 31. Prefer numeric API levels in new code. */
#define __ANDROID_API_S__ 31

/** Deprecated name for API level 33. Prefer numeric API levels in new code. */
#define __ANDROID_API_T__ 33

/** Deprecated name for API level 34. Prefer numeric API levels in new code. */
#define __ANDROID_API_U__ 34

/** Deprecated name for API level 35. Prefer numeric API levels in new code. */
#define __ANDROID_API_V__ 35

/* This file is included in <features.h>, and might be used from .S files. */
#if !defined(__ASSEMBLY__)

/**
 * Returns the `targetSdkVersion` of the caller, or `__ANDROID_API_FUTURE__` if
 * there is no known target SDK version (for code not running in the context of
 * an app).
 *
 * The returned values correspond to the named constants in `<android/api-level.h>`,
 * and is equivalent to the AndroidManifest.xml `targetSdkVersion`.
 *
 * See also android_get_device_api_level().
 *
 * Available since API level 24.
 */
#if __BIONIC_AVAILABILITY_GUARD(24)
int android_get_application_target_sdk_version() __INTRODUCED_IN(24);
#endif /* __BIONIC_AVAILABILITY_GUARD(24) */


#if __ANDROID_API__ < 29

/* android_get_device_api_level is a static inline before API level 29. */
#define __BIONIC_GET_DEVICE_API_LEVEL_INLINE static __inline
#include <bits/get_device_api_level_inlines.h>
#undef __BIONIC_GET_DEVICE_API_LEVEL_INLINE

#else

/**
 * Returns the API level of the device we're actually running on, or -1 on failure.
 * The returned values correspond to the named constants in `<android/api-level.h>`,
 * and is equivalent to the Java `Build.VERSION.SDK_INT` API.
 *
 * See also android_get_application_target_sdk_version().
 *
 * Available since API level 29.
 */
int android_get_device_api_level() __INTRODUCED_IN(29);

#endif

#endif /* defined(__ASSEMBLY__) */

__END_DECLS

/** @} */
