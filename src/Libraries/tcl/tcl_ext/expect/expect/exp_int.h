/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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
#ifndef _EXPECT_INT_H
#define _EXPECT_INT_H

#ifndef TRUE
#define FALSE 0
#define TRUE 1
#endif

#ifndef HAVE_MEMCPY
#define memcpy(x,y,len) bcopy(y,x,len)
#endif

#include <tcl.h>
#include <errno.h>

void	exp_console_set     _ANSI_ARGS_((void));
void	expDiagLogPtrSet    _ANSI_ARGS_((void (*)_ANSI_ARGS_((char *))));
void	expDiagLogPtr       _ANSI_ARGS_((char *));
void	expDiagLogPtrX      _ANSI_ARGS_((char *,int));
void	expDiagLogPtrStr    _ANSI_ARGS_((char *,char *));
void	expDiagLogPtrStrStr _ANSI_ARGS_((char *,char *,char *));
void	expErrnoMsgSet      _ANSI_ARGS_((char * (*) _ANSI_ARGS_((int))));
char * expErrnoMsg    _ANSI_ARGS_((int));

#ifdef NO_STDLIB_H
#  include "../compat/stdlib.h"
#else
#  include <stdlib.h>		/* for malloc */
#endif /*NO_STDLIB_H*/

#endif /* _EXPECT_INT_H */
