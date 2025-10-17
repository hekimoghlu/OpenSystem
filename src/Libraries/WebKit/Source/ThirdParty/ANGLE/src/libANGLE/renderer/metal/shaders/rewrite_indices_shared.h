/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

//
// Copyright 2021 The ANGLE Project. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// rewrite_incides_shared:
// defines for provoking vertex rewriting.

#ifndef rewrite_indices_shared_h
#define rewrite_indices_shared_h
#define MTL_FIX_INDEX_BUFFER_KEY_FUNCTION_CONSTANT_INDEX 1

#define MtlFixIndexBufferKeyTypeMask 0x03U
#define MtlFixIndexBufferKeyInShift 0U
#define MtlFixIndexBufferKeyOutShift 2U
#define MtlFixIndexBufferKeyVoid 0U
#define MtlFixIndexBufferKeyUint16 2U
#define MtlFixIndexBufferKeyUint32 3U
#define MtlFixIndexBufferKeyModeMask 0x0FU
#define MtlFixIndexBufferKeyModeShift 4U
#define MtlFixIndexBufferKeyPoints 0x00U
#define MtlFixIndexBufferKeyLines 0x01U
#define MtlFixIndexBufferKeyLineLoop 0x02U
#define MtlFixIndexBufferKeyLineStrip 0x03U
#define MtlFixIndexBufferKeyTriangles 0x04U
#define MtlFixIndexBufferKeyTriangleStrip 0x05U
#define MtlFixIndexBufferKeyTriangleFan 0x06U
#define MtlFixIndexBufferKeyPrimRestart 0x00100U
#define MtlFixIndexBufferKeyProvokingVertexLast 0x00200U
#endif /* rewrite_indices_shared_h */
