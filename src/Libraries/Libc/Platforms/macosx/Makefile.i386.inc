/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

#
# Selectable features for MacOSX
#

# Legacy *64 APIs
FEATURE_LEGACY_64_APIS = 1

# Legacy crt1.o environ support
FEATURE_LEGACY_CRT1_ENVIRON = 1

# Legacy rune APIs
FEATURE_LEGACY_RUNE_APIS = 1

# Legacy utmp APIs
FEATURE_LEGACY_UTMP_APIS = 1

# OSThermalNotification APIs
FEATURE_THERM_NOTIFICATION_APIS = 1

# No pre-1050 variants (should match sys/cdefs.h)
FEATURE_ONLY_1050_VARIANTS = 0

# No legacy variants (should match sys/cdefs.h)
FEATURE_ONLY_UNIX_CONFORMANCE = 0

# Only 64-bit ino_t (should match sys/cdefs.h)
FEATURE_ONLY_64_BIT_INO_T = 0

# Patch 3333969
FEATURE_PATCH_3333969 = 1

# Patch 3417676
FEATURE_PATCH_3417676 = 1

# plockstat dtrace support
FEATURE_PLOCKSTAT = 1

# Timezone change notification
FEATURE_TIMEZONE_CHANGE_NOTIFICATION = 1

# Extensible printf performance enhancement (uses more memory)
FEATURE_XPRINTF_PERF = 1

# Disable registration of specific signals (<rdar://problem/21952708>)
# FEATURE_SIGNAL_RESTRICTION = 0

# Enable 32-bit compilation in unistd.h (<rdar://problem/51818745>)
# FEATURE_POSIX_ILP32_ALLOW = 0
