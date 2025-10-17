/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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
#ifndef VPX_ARGS_H_
#define VPX_ARGS_H_
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct arg {
  char **argv;
  const char *name;
  const char *val;
  unsigned int argv_step;
  const struct arg_def *def;
};

struct arg_enum_list {
  const char *name;
  int val;
};
#define ARG_ENUM_LIST_END \
  { 0 }

typedef struct arg_def {
  const char *short_name;
  const char *long_name;
  int has_val;
  const char *desc;
  const struct arg_enum_list *enums;
} arg_def_t;
#define ARG_DEF(s, l, v, d) \
  { s, l, v, d, NULL }
#define ARG_DEF_ENUM(s, l, v, d, e) \
  { s, l, v, d, e }
#define ARG_DEF_LIST_END \
  { 0 }

int arg_match(struct arg *arg_, const struct arg_def *def, char **argv);
const char *arg_next(struct arg *arg);
void arg_show_usage(FILE *fp, const struct arg_def *const *defs);
char **argv_dup(int argc, const char **argv);

// Note: arg_match() must be called before invoking these functions.
unsigned int arg_parse_uint(const struct arg *arg);
int arg_parse_int(const struct arg *arg);
struct vpx_rational arg_parse_rational(const struct arg *arg);
int arg_parse_enum(const struct arg *arg);
int arg_parse_enum_or_int(const struct arg *arg);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_ARGS_H_
