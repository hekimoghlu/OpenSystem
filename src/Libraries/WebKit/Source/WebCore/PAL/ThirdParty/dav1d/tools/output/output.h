/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#ifndef DAV1D_OUTPUT_OUTPUT_H
#define DAV1D_OUTPUT_OUTPUT_H

#include "picture.h"

typedef struct MuxerContext MuxerContext;

int output_open(MuxerContext **c, const char *name, const char *filename,
                const Dav1dPictureParameters *p, const unsigned fps[2]);
int output_write(MuxerContext *ctx, Dav1dPicture *pic);
void output_close(MuxerContext *ctx);
/**
 * Verifies the muxed data (for example in the md5 muxer). Replaces output_close.
 *
 * @param  hash_string Muxer specific reference value.
 *
 * @return 0 on success.
 */
int output_verify(MuxerContext *ctx, const char *hash_string);

#endif /* DAV1D_OUTPUT_OUTPUT_H */
