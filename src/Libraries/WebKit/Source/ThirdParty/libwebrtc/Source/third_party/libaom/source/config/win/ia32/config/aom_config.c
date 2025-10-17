/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#include "aom/aom_codec.h"
static const char* const cfg = "cmake ../source/libaom -G \"Unix Makefiles\" -DCMAKE_TOOLCHAIN_FILE=\"../source/libaom/build/cmake/toolchains/i686-linux-gcc.cmake\" -DAOM_RTCD_FLAGS=--require-mmx;--require-sse;--require-sse2 -DCONFIG_AV1_DECODER=0 -DCONFIG_AV1_ENCODER=1 -DCONFIG_LIBYUV=0 -DCONFIG_AV1_HIGHBITDEPTH=0 -DCONFIG_AV1_TEMPORAL_DENOISING=1 -DCONFIG_PIC=1 -DCONFIG_QUANT_MATRIX=0 -DCONFIG_REALTIME_ONLY=1 -DCONFIG_SIZE_LIMIT=1 -DDECODE_HEIGHT_LIMIT=16384 -DDECODE_WIDTH_LIMIT=16384";
const char *aom_codec_build_config(void) {return cfg;}
