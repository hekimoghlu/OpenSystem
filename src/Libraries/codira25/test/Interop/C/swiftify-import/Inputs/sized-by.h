/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#include <stdint.h>

#ifndef __sized_by
#define __sized_by(x) __attribute__((__sized_by__(x)))
#endif

void simple(int len, void * __sized_by(len) p);

void languageAttr(int len, void *p) __attribute__((language_attr(
    "@_CodiraifyImport(.sizedBy(pointer: .param(2), size: \"len\"))")));

void shared(int len, void * __sized_by(len) p1, void * __sized_by(len) p2);

void complexExpr(int len, int offset, void * __sized_by(len - offset) p);

void nullUnspecified(int len, void * __sized_by(len) _Null_unspecified p);

void nonnull(int len, void * __sized_by(len) _Nonnull p);

void nullable(int len, void * __sized_by(len) _Nullable p);

void * __sized_by(len) returnPointer(int len);

typedef struct foo opaque_t;
void opaque(int len, opaque_t * __sized_by(len) p);

typedef opaque_t *opaqueptr_t;
void opaqueptr(int len, opaqueptr_t __sized_by(len) p);

void charsized(char *__sized_by(size), int size);

uint8_t *__sized_by(size) bytesized(int size);

void doublebytesized(uint16_t *__sized_by(size), int size);

typedef uint8_t * bytesizedptr_t;
void aliasedBytesized(bytesizedptr_t __sized_by(size) p, int size);
