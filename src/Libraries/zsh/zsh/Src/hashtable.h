/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
/* Builtin function numbers; used by handler functions that handle more *
 * than one builtin.  Note that builtins such as compctl, that are not  *
 * overloaded, don't get a number.                                      */

#define BIN_TYPESET   0
#define BIN_BG        1
#define BIN_FG        2
#define BIN_JOBS      3
#define BIN_WAIT      4
#define BIN_DISOWN    5
#define BIN_BREAK     6
#define BIN_CONTINUE  7
#define BIN_EXIT      8
#define BIN_RETURN    9
#define BIN_CD       10
#define BIN_POPD     11
#define BIN_PUSHD    12
#define BIN_PRINT    13
#define BIN_EVAL     14
#define BIN_SCHED    15
#define BIN_FC       16
#define BIN_R	     17
#define BIN_PUSHLINE 18
#define BIN_LOGOUT   19
#define BIN_TEST     20
#define BIN_BRACKET  21
#define BIN_READONLY 22
#define BIN_ECHO     23
#define BIN_DISABLE  24
#define BIN_ENABLE   25
#define BIN_PRINTF   26
#define BIN_COMMAND  27
#define BIN_UNHASH   28
#define BIN_UNALIAS  29
#define BIN_UNFUNCTION  30
#define BIN_UNSET    31
#define BIN_EXPORT   32

/* These currently depend on being 0 and 1. */
#define BIN_SETOPT    0
#define BIN_UNSETOPT  1
