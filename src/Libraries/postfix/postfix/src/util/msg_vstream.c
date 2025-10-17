/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
/* System libraries. */

#include <sys_defs.h>
#include <errno.h>
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>

/* Utility library. */

#include "vstream.h"
#include "msg.h"
#include "msg_output.h"
#include "msg_vstream.h"

 /*
  * Private state.
  */
static const char *msg_tag;
static VSTREAM *msg_stream;

/* msg_vstream_print - log diagnostic to VSTREAM */

static void msg_vstream_print(int level, const char *text)
{
    static const char *level_text[] = {
	"info", "warning", "error", "fatal", "panic",
    };

    if (level < 0 || level >= (int) (sizeof(level_text) / sizeof(level_text[0])))
	msg_panic("invalid severity level: %d", level);
    if (level == MSG_INFO) {
	vstream_fprintf(msg_stream, "%s: %s\n",
			msg_tag, text);
    } else {
	vstream_fprintf(msg_stream, "%s: %s: %s\n",
			msg_tag, level_text[level], text);
    }
    vstream_fflush(msg_stream);
}

/* msg_vstream_init - initialize */

void    msg_vstream_init(const char *name, VSTREAM *vp)
{
    static int first_call = 1;

    msg_tag = name;
    msg_stream = vp;
    if (first_call) {
	first_call = 0;
	msg_output(msg_vstream_print);
    }
}
