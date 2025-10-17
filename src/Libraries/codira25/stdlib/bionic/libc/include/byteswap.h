/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
 * @file byteswap.h
 * @brief Byte-swapping macros.
 */

#include <sys/cdefs.h>
#include <sys/endian.h>

/**
 * [bswap_16(3)](https://man7.org/linux/man-pages/man3/bswap_16.3.html) swaps the bytes in a
 * 16-bit value.
 */
#define bswap_16(x) __swap16(x)

/**
 * [bswap_32(3)](https://man7.org/linux/man-pages/man3/bswap_32.3.html) swaps the bytes in a
 * 32-bit value.
 */
#define bswap_32(x) __swap32(x)

/**
 * [bswap_64(3)](https://man7.org/linux/man-pages/man3/bswap_64.3.html) swaps the bytes in a
 * 64-bit value.
 */
#define bswap_64(x) __swap64(x)
