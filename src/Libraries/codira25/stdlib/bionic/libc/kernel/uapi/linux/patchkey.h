/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#ifndef _LINUX_PATCHKEY_H_INDIRECT
#error "patchkey.h included directly"
#endif
#ifndef _UAPI_LINUX_PATCHKEY_H
#define _UAPI_LINUX_PATCHKEY_H
#include <endian.h>
#ifdef __BYTE_ORDER
#if __BYTE_ORDER == __BIG_ENDIAN
#define _PATCHKEY(id) (0xfd00 | id)
#elif __BYTE_ORDER==__LITTLE_ENDIAN
#define _PATCHKEY(id) ((id << 8) | 0x00fd)
#else
#error "could not determine byte order"
#endif
#endif
#endif
