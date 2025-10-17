/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
// FUNCTION_VARIANT_PER_PROCESS(_keyIndex, _name, _runtimeCheck)
//


//
//  per-process function variants
//
FUNCTION_VARIANT_PER_PROCESS(0, "default", true)

// These were added in Spring 2025
FUNCTION_VARIANT_PER_PROCESS(1, "translated",   config.process.isTranslated)
FUNCTION_VARIANT_PER_PROCESS(3, "no_overread", (config.process.environ("LIBPLATFORM_NO_OVERREAD") != nullptr))

// These were added in Fall 2025

