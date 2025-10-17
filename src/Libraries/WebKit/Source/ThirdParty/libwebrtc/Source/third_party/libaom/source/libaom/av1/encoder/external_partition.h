/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#ifndef AOM_AV1_ENCODER_EXTERNAL_PARTITION_H_
#define AOM_AV1_ENCODER_EXTERNAL_PARTITION_H_

#include <stdbool.h>

#include "aom/aom_codec.h"
#include "aom/aom_external_partition.h"
#include "config/aom_config.h"

#ifdef __cplusplus
extern "C" {
#endif
/*!\cond */

typedef struct ExtPartController {
  int ready;
  int test_mode;
  aom_ext_part_config_t config;
  aom_ext_part_model_t model;
  aom_ext_part_funcs_t funcs;
} ExtPartController;

aom_codec_err_t av1_ext_part_create(aom_ext_part_funcs_t funcs,
                                    aom_ext_part_config_t config,
                                    ExtPartController *ext_part_controller);

aom_codec_err_t av1_ext_part_delete(ExtPartController *ext_part_controller);

bool av1_ext_part_get_partition_decision(ExtPartController *ext_part_controller,
                                         aom_partition_decision_t *decision);

bool av1_ext_part_send_features(ExtPartController *ext_part_controller,
                                const aom_partition_features_t *features);

#if CONFIG_PARTITION_SEARCH_ORDER
bool av1_ext_part_send_partition_stats(ExtPartController *ext_part_controller,
                                       const aom_partition_stats_t *stats);

aom_ext_part_decision_mode_t av1_get_ext_part_decision_mode(
    const ExtPartController *ext_part_controller);
#endif  // CONFIG_PARTITION_SEARCH_ORDER

/*!\endcond */
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_EXTERNAL_PARTITION_H_
