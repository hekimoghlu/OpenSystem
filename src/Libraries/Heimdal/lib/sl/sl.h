/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
/* $Id$ */

#ifndef _SL_H
#define _SL_H

#define SL_BADCOMMAND -1

typedef int (*cmd_func)(int, char **);

struct sl_cmd {
  const char *name;
  cmd_func func;
  const char *usage;
  const char *help;
};

typedef struct sl_cmd SL_cmd;

#ifdef __cplusplus
extern "C" {
#endif

void sl_help (SL_cmd *, int argc, char **argv);
int  sl_loop (SL_cmd *, const char *prompt);
int  sl_command_loop (SL_cmd *cmds, const char *prompt, void **data);
int  sl_command (SL_cmd *cmds, int argc, char **argv);
int sl_make_argv(char*, int*, char***);
void sl_apropos (SL_cmd *cmd, const char *topic);
SL_cmd *sl_match (SL_cmd *cmds, char *cmd, int exactp);
void sl_slc_help (SL_cmd *cmds, int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* _SL_H */
