/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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

//  Tunables
#define LZ4_COMPRESS_HASH_BITS 10
#define LZ4_COMPRESS_HASH_ENTRIES (1 << LZ4_COMPRESS_HASH_BITS)
#define LZ4_COMPRESS_HASH_MULTIPLY 2654435761U
#define LZ4_COMPRESS_HASH_SHIFT (32 - LZ4_COMPRESS_HASH_BITS)

//  Not tunables
#define LZ4_GOFAST_SAFETY_MARGIN 128
#define LZ4_DISTANCE_BOUND 65536
