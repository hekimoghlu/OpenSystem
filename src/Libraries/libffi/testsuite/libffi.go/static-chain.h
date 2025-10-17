/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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

#ifdef __aarch64__
# define STATIC_CHAIN_REG  "x18"
#elif defined(__alpha__)
# define STATIC_CHAIN_REG  "$1"
#elif defined(__arm__)
# define STATIC_CHAIN_REG  "ip"
#elif defined(__sparc__)
# if defined(__arch64__) || defined(__sparcv9)
#  define STATIC_CHAIN_REG "g5"
# else
#  define STATIC_CHAIN_REG "g2"
# endif
#elif defined(__x86_64__)
# define STATIC_CHAIN_REG  "r10"
#elif defined(__i386__)
# ifndef ABI_NUM
#  define STATIC_CHAIN_REG  "ecx"	/* FFI_DEFAULT_ABI only */
# endif
#endif
