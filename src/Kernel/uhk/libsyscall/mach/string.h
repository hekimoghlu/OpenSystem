/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#ifndef _STRING_H_
#define _STRING_H_

#include <stdarg.h>
#include <_types.h>
#include <sys/_types/_null.h>
#include <sys/_types/_size_t.h>
#include <sys/_types/_uintptr_t.h>

// We're purposefully called "string.h" in order to superceed any use
// of Libc's string.h (which no one should be using bar MIG) in order
// to override their use of memcpy.

int _mach_snprintf(char *buffer, int length, const char *fmt, ...) __printflike(3, 4);
int _mach_vsnprintf(char *buffer, int length, const char *fmt, va_list ap) __printflike(3, 0);

// These declarations are just for MIG, other users should include string/strings.h
// These symbols are defined in _libc_funcptr.c

void *memcpy(void *dst0, const void *src0, size_t length);
void *memset(void *dst0, int c0, size_t length);
void bzero(void *dst0, size_t length);

#endif /* _STRING_H_ */
