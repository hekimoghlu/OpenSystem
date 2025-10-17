/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#include "expect_cf.h"
#include <stdio.h>

#include "tcl.h"
#include "exp_tty_in.h"
#include "exp_rename.h"
#include "expect_comm.h"
#include "exp_command.h"
#include "exp_log.h"

#ifndef TRUE
#define FALSE 0
#define TRUE 1
#endif

int exp_disconnected = FALSE;		/* not disc. from controlling tty */

/*static*/ exp_tty exp_tty_current, exp_tty_cooked;
#define tty_current exp_tty_current
#define tty_cooked exp_tty_cooked

void
exp_init_tty()
{
	extern exp_tty exp_tty_original;

	/* save original user tty-setting in 'cooked', just in case user */
	/* asks for it without earlier telling us what cooked means to them */
	tty_cooked = exp_tty_original;

	/* save our current idea of the terminal settings */
	tty_current = exp_tty_original;
}

