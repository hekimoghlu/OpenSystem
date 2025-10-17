/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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

//  Common header to enable/disable the assembly code paths
//  Rule: one define for each assembly source file

//  To enable assembly
#if defined __arm64__
#define LZ4_ENABLE_ASSEMBLY_ENCODE_ARM64 1
#define LZ4_ENABLE_ASSEMBLY_DECODE_ARM64 1
#elif defined __ARM_NEON__
#define LZ4_ENABLE_ASSEMBLY_ENCODE_ARMV7 1
#define LZ4_ENABLE_ASSEMBLY_DECODE_ARMV7 1
#elif defined __x86_64__
#define LZ4_ENABLE_ASSEMBLY_DECODE_X86_64 1
#endif

//  To disable C
#define LZ4_ENABLE_ASSEMBLY_ENCODE ((LZ4_ENABLE_ASSEMBLY_ENCODE_ARMV7) || (LZ4_ENABLE_ASSEMBLY_ENCODE_ARM64))
#define LZ4_ENABLE_ASSEMBLY_DECODE (LZ4_ENABLE_ASSEMBLY_DECODE_ARM64 || LZ4_ENABLE_ASSEMBLY_DECODE_ARMV7 || LZ4_ENABLE_ASSEMBLY_DECODE_X86_64)
