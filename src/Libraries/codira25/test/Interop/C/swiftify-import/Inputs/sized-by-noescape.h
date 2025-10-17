/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#define __noescape __attribute__((noescape))

void simple(int len, const void * __sized_by(len) __noescape p);

void languageAttr(int len, const void *p) __attribute__((language_attr(
    "@_CodiraifyImport(.sizedBy(pointer: .param(2), size: \"len\"), .nonescaping(pointer: .param(2)), spanAvailability: \"visionOS 1.0, tvOS 12.2, watchOS 5.2, iOS 12.2, macOS 10.14.4\")")));

void shared(int len, const void * __sized_by(len) __noescape p1, const void * __sized_by(len) __noescape p2);

void complexExpr(int len, int offset, const void * __sized_by(len - offset) __noescape p);

void nullUnspecified(int len, const void * __sized_by(len) __noescape _Null_unspecified p);

void nonnull(int len, const void * __sized_by(len) __noescape _Nonnull p);

void nullable(int len, const void * __sized_by(len) __noescape _Nullable p);

const void * __sized_by(len) __noescape _Nonnull returnPointer(int len);

typedef struct foo opaque_t;
void opaque(int len, opaque_t * __sized_by(len) __noescape p);

void bytesized(int size, const uint8_t *__sized_by(size) __noescape);

void charsized(char *__sized_by(size) __noescape, int size);

void doublebytesized(uint16_t *__sized_by(size) __noescape, int size);
