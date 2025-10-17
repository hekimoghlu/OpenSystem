/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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

void begin();

// line comment
void lineComment(int len, int * __counted_by(len) p);

/// line doc comment
/// 
/// Here's a more complete description.
///
/// @param len the buffer length
/// @param p the buffer
void lineDocComment(int len, int * __counted_by(len) p);

/*
 block comment
 */
void blockComment(int len, int * __counted_by(len) p);

/**
 * block doc comment
 * 
 * NB: it's very important to pass the correct length to this function
 * @param len don't mess this one up
 * @param p   some integers to play with
 */
void blockDocComment(int len, int * __counted_by(len) p);
