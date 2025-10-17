/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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

#ifdef __APPLE__
#include <Availability.h>
#include <AvailabilityMacros.h>
#include <TargetConditionals.h>

#if (defined(TARGET_IPHONE_SIMULATOR) && TARGET_IPHONE_SIMULATOR)
#define HAVE_VTB_REQUIREDLOWLATENCY 0
#define ENABLE_VCP_FOR_H264_BASELINE 0
#define ENABLE_H264_HIGHPROFILE_AUTOLEVEL 0
#elif (defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE) || (defined(TARGET_OS_MAC) && TARGET_OS_MAC)
#define HAVE_VTB_REQUIREDLOWLATENCY 1
#define ENABLE_VCP_FOR_H264_BASELINE 1
#define ENABLE_H264_HIGHPROFILE_AUTOLEVEL 1
#endif

#if (defined(TARGET_OS_MAC) && TARGET_OS_MAC)
#define ENABLE_LOW_LATENCY_INTEL_ENCODER_FOR_LOW_RESOLUTION (__MAC_OS_X_VERSION_MIN_REQUIRED > 150000)
#else
#define ENABLE_LOW_LATENCY_INTEL_ENCODER_FOR_LOW_RESOLUTION 1
#endif

#endif // __APPLE__
