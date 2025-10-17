/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
// included by:
//   ld to convert function variant names to keyIndexes
//   dyld to drive which function to run to set keyIndex bool
//
// FUNCTION_VARIANT_SYSTEM_WIDE(_keyIndex, _name, _runtimeCheck)
//


//
//  shared function variants (same for all processes)
//
FUNCTION_VARIANT_SYSTEM_WIDE(0, "default",           true)

// These were added in Spring 2025
FUNCTION_VARIANT_SYSTEM_WIDE(1, "internalInstall",   config.security.internalInstall)
FUNCTION_VARIANT_SYSTEM_WIDE(2, "customerInstall",   !config.security.internalInstall)
FUNCTION_VARIANT_SYSTEM_WIDE(3, "lockdown",           config.security.lockdownMode)

// These were added in Fall 2025




