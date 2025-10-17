/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#ifndef AOM_AV1_COMMON_DEBUGMODES_H_
#define AOM_AV1_COMMON_DEBUGMODES_H_

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/enums.h"

void av1_print_modes_and_motion_vectors(AV1_COMMON *cm, const char *file);
void av1_print_uncompressed_frame_header(const uint8_t *data, int size,
                                         const char *filename);
void av1_print_frame_contexts(const FRAME_CONTEXT *fc, const char *filename);

#endif  // AOM_AV1_COMMON_DEBUGMODES_H_
