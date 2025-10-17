/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#ifndef AOM_AOM_DSP_VMAF_H_
#define AOM_AOM_DSP_VMAF_H_

#include <libvmaf/libvmaf.h>
#include <stdbool.h>

#include "aom_scale/yv12config.h"

void aom_init_vmaf_context(VmafContext **vmaf_context, VmafModel *vmaf_model,
                           bool cal_vmaf_neg);
void aom_close_vmaf_context(VmafContext *vmaf_context);

void aom_init_vmaf_model(VmafModel **vmaf_model, const char *model_path);
void aom_close_vmaf_model(VmafModel *vmaf_model);

void aom_calc_vmaf(VmafModel *vmaf_model, const YV12_BUFFER_CONFIG *source,
                   const YV12_BUFFER_CONFIG *distorted, int bit_depth,
                   bool cal_vmaf_neg, double *vmaf);

void aom_read_vmaf_image(VmafContext *vmaf_context,
                         const YV12_BUFFER_CONFIG *source,
                         const YV12_BUFFER_CONFIG *distorted, int bit_depth,
                         int frame_index);

double aom_calc_vmaf_at_index(VmafContext *vmaf_context, VmafModel *vmaf_model,
                              int frame_index);

void aom_flush_vmaf_context(VmafContext *vmaf_context);

#endif  // AOM_AOM_DSP_VMAF_H_
