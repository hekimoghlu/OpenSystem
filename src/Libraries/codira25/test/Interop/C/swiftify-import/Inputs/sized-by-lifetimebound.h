/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
#define __lifetimebound __attribute__((lifetimebound))

const void * __sized_by(len) simple(int len, int len2, const void * __sized_by(len2) __lifetimebound p);

const void * __sized_by(len) shared(int len, const void * __sized_by(len) __lifetimebound p);

const void * __sized_by(len - offset) complexExpr(int len, int offset, int len2, const void * __sized_by(len2) __lifetimebound p);

const void * __sized_by(len) _Null_unspecified nullUnspecified(int len, int len2, const void * __sized_by(len2) __lifetimebound _Null_unspecified p);

const void * __sized_by(len) _Nonnull nonnull(int len, int len2, const void * __sized_by(len2) __lifetimebound _Nonnull p);

const void * __sized_by(len) _Nullable nullable(int len, int len2, const void * __sized_by(len2) __lifetimebound _Nullable p);

typedef struct foo opaque_t;
opaque_t * __sized_by(len) opaque(int len, int len2, opaque_t * __sized_by(len2) __lifetimebound p);

const void * __sized_by(len) nonsizedLifetime(int len, const void * __lifetimebound p);

uint8_t *__sized_by(size)  bytesized(int size, const uint8_t *__sized_by(size) __lifetimebound);

char *__sized_by(size) charsized(char *__sized_by(size) __lifetimebound, int size);

const uint16_t *__sized_by(size)  doublebytesized(uint16_t *__sized_by(size) __lifetimebound, int size);
