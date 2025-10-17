/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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

#define __counted_by(x) __attribute__((__counted_by__(x)))

void simple(int len, int * __counted_by(len) p);

void simpleFlipped(int * __counted_by(len) p, int len);

void languageAttr(int len, int *p) __attribute__((
    language_attr("@_CodiraifyImport(.countedBy(pointer: .param(2), count: \"len\"))")));

void shared(int len, int * __counted_by(len) p1, int * __counted_by(len) p2);

void complexExpr(int len, int offset, int * __counted_by(len - offset) p);

void nullUnspecified(int len, int * __counted_by(len) _Null_unspecified p);

void nonnull(int len, int * __counted_by(len) _Nonnull p);

void nullable(int len, int * __counted_by(len) _Nullable p);

int * __counted_by(len) returnPointer(int len);

void offByOne(int len, int * __counted_by(len + 1) p);

void offBySome(int len, int offset, int * __counted_by(len + (1 + offset)) p);

void scalar(int m, int n, int * __counted_by(m * n) p);

void bitwise(int m, int n, int o, int * __counted_by(m & n | ~o) p);

void bitshift(int m, int n, int o, int * __counted_by(m << (n >> o)) p);

void constInt(int * __counted_by(42 * 10) p);

void constFloatCastedToInt(int * __counted_by((int) (4.2 / 12)) p);

void sizeofType(int * __counted_by(sizeof(int *)) p);

void sizeofParam(int * __counted_by(sizeof(p)) p);

void derefLen(int * len, int * __counted_by(*len) p);

void lNot(int len, int * __counted_by(!len) p);

void lAnd(int len, int * __counted_by(len && len) p);

void lOr(int len, int * __counted_by(len || len) p);

void floatCastToInt(float meters, int * __counted_by((int) meters) p);

void pointerCastToInt(int *square, int * __counted_by((int) square) p);

void nanAsInt(int * __counted_by((int) (0 / 0)) p);

void unsignedLiteral(int * __counted_by(2u) p);

void longLiteral(int * __counted_by(2l) p);

void hexLiteral(int * __counted_by(0xfa) p);

void binaryLiteral(int * __counted_by(0b10) p);

void octalLiteral(int * __counted_by(0777) p);
