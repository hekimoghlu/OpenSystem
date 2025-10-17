/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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
#ifndef _WAIT_H
#define _WAIT_H

#ifndef _TYPES_H		/* not quite right */
#include <sys/types.h>
#endif

#define __LOW(v)	((v) & 0377)
#define __HIGH(v)	(((v) >> 8) & 0377)

#define WNOHANG         1	/* do not wait for child to exit */
#define WUNTRACED       2	/* for job control; not implemented */

#define WIFEXITED(s)	(__LOW(s) == 0)		    /* normal exit */
#define WEXITSTATUS(s)	(__HIGH(s))			    /* exit status */
#define WTERMSIG(s)	(__LOW(s) & 0177)		    /* sig value */
#define WIFSIGNALED(s)	((((unsigned int)(s)-1) & 0xFFFF) < 0xFF) /* signaled */
#define WIFSTOPPED(s)	(__LOW(s) == 0177)		    /* stopped */
#define WSTOPSIG(s)	(__HIGH(s) & 0377)		    /* stop signal */

/* Function Prototypes. */
#ifndef _ANSI_H
#include <ansi.h>
#endif

_PROTOTYPE( pid_t wait, (int *_stat_loc)			   	   );
_PROTOTYPE( pid_t waitpid, (pid_t _pid, int *_stat_loc, int _options)	   );

#endif /* _WAIT_H */
