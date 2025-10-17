/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#ifndef __EXP_TTY_H__
#define __EXP_TTY_H__

#include "expect_cf.h"

extern int exp_dev_tty;
extern int exp_ioctled_devtty;
extern int exp_stdin_is_tty;
extern int exp_stdout_is_tty;

void exp_tty_raw(int set);
void exp_tty_echo(int set);
void exp_tty_break(Tcl_Interp *interp, int fd);
int exp_tty_raw_noecho(Tcl_Interp *interp, exp_tty *tty_old, int *was_raw, int *was_echo);
int exp_israw(void);
int exp_isecho(void);

void exp_tty_set(Tcl_Interp *interp, exp_tty *tty, int raw, int echo);
int exp_tty_set_simple(exp_tty *tty);
int exp_tty_get_simple(exp_tty *tty);

#endif	/* __EXP_TTY_H__ */
