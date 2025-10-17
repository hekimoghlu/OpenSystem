/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#define IS_ASCII_OCTET(c)   (((c) & 0x80) == 0)
#define IS_UTF8_TRAIL(c)    (((c) & 0xC0) == 0x80)
#define IS_UTF8_LEAD2(c)    (((c) & 0xE0) == 0xC0)
#define IS_UTF8_LEAD3(c)    (((c) & 0xF0) == 0xE0)
#define IS_UTF8_LEAD4(c)    (((c) & 0xF8) == 0xF0)
#define IS_UTF8_LEAD5(c)    (((c) & 0xFC) == 0xF8)
#define IS_UTF8_LEAD6(c)    (((c) & 0xFE) == 0xFC)
#define IS_UTF8_INVALID(c)  (((c) & 0xFE) == 0xFE)
#define IS_UTF8_LEAD(c)     (((c) & 0xC0) == 0xC0 && !IS_UTF8_INVALID(c))
