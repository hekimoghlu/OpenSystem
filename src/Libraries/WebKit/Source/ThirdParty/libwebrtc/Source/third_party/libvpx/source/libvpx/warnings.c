/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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
#include "./warnings.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vpx_encoder.h"

#include "./tools_common.h"
#include "./vpxenc.h"

static const char quantizer_warning_string[] =
    "Bad quantizer values. Quantizer values should not be equal, and should "
    "differ by at least 8.";
static const char lag_in_frames_with_realtime[] =
    "Lag in frames is ignored when deadline is set to realtime for cbr mode.";

struct WarningListNode {
  const char *warning_string;
  struct WarningListNode *next_warning;
};

struct WarningList {
  struct WarningListNode *warning_node;
};

static void add_warning(const char *warning_string,
                        struct WarningList *warning_list) {
  struct WarningListNode **node = &warning_list->warning_node;

  struct WarningListNode *new_node = malloc(sizeof(*new_node));
  if (new_node == NULL) {
    fatal("Unable to allocate warning node.");
  }

  new_node->warning_string = warning_string;
  new_node->next_warning = NULL;

  while (*node != NULL) node = &(*node)->next_warning;

  *node = new_node;
}

static void free_warning_list(struct WarningList *warning_list) {
  while (warning_list->warning_node != NULL) {
    struct WarningListNode *const node = warning_list->warning_node;
    warning_list->warning_node = node->next_warning;
    free(node);
  }
}

static int continue_prompt(int num_warnings) {
  int c;
  fprintf(stderr,
          "%d encoder configuration warning(s). Continue? (y to continue) ",
          num_warnings);
  c = getchar();
  return c == 'y';
}

static void check_quantizer(int min_q, int max_q,
                            struct WarningList *warning_list) {
  const int lossless = min_q == 0 && max_q == 0;
  if (!lossless && (min_q == max_q || abs(max_q - min_q) < 8))
    add_warning(quantizer_warning_string, warning_list);
}

static void check_lag_in_frames_realtime_deadline(
    int lag_in_frames, int deadline, int rc_end_usage,
    struct WarningList *warning_list) {
  if (deadline == VPX_DL_REALTIME && lag_in_frames != 0 && rc_end_usage == 1)
    add_warning(lag_in_frames_with_realtime, warning_list);
}

void check_encoder_config(int disable_prompt,
                          const struct VpxEncoderConfig *global_config,
                          const struct vpx_codec_enc_cfg *stream_config) {
  int num_warnings = 0;
  struct WarningListNode *warning = NULL;
  struct WarningList warning_list = { 0 };

  check_quantizer(stream_config->rc_min_quantizer,
                  stream_config->rc_max_quantizer, &warning_list);
  check_lag_in_frames_realtime_deadline(
      stream_config->g_lag_in_frames, global_config->deadline,
      stream_config->rc_end_usage, &warning_list);
  /* Count and print warnings. */
  for (warning = warning_list.warning_node; warning != NULL;
       warning = warning->next_warning, ++num_warnings) {
    warn("%s", warning->warning_string);
  }

  free_warning_list(&warning_list);

  if (num_warnings) {
    if (!disable_prompt && !continue_prompt(num_warnings)) exit(EXIT_FAILURE);
  }
}
