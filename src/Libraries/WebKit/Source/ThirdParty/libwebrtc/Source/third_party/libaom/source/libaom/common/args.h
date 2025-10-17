/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#ifndef AOM_COMMON_ARGS_H_
#define AOM_COMMON_ARGS_H_
#include <stdio.h>

#include "aom/aom_codec.h"
#include "aom/aom_encoder.h"
#include "common/args_helper.h"

#ifdef __cplusplus
extern "C" {
#endif

int arg_match(struct arg *arg_, const struct arg_def *def, char **argv);
int parse_cfg(const char *file, cfg_options_t *config);
const char *arg_next(struct arg *arg);
void arg_show_usage(FILE *fp, const struct arg_def *const *defs);
char **argv_dup(int argc, const char **argv);

unsigned int arg_parse_uint(const struct arg *arg);
int arg_parse_int(const struct arg *arg);
struct aom_rational arg_parse_rational(const struct arg *arg);
int arg_parse_enum(const struct arg *arg);
int arg_parse_enum_or_int(const struct arg *arg);
int arg_parse_list(const struct arg *arg, int *list, int n);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_ARGS_H_
