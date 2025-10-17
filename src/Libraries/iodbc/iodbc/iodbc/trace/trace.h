/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#ifndef _IODBCDM_TRACE_H
#define _IODBCDM_TRACE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <pwd.h>
#include <unistd.h>

#include <sql.h>
#include <sqlext.h>
#include <sqlucode.h>

#include "herr.h"
#include "henv.h"
#include "ithread.h"
#include "unicode.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  Useful constants and macros
 */
#define MAX_EMIT_STRING		40000L		/* = 1000 lines in output */
#define MAX_EMIT_BINARY		10000L		/* = 1000 lines in output */

#define TRACE_ENTER	0
#define TRACE_LEAVE	1

#undef  _S
#define _S(X)	case X: ptr = #X; break

/*
 *  Is the argument and input or output parameter or both
 */
#define TRACE_NEVER		(0)
#define TRACE_INPUT		(trace_leave == TRACE_ENTER)
#define TRACE_OUTPUT		(trace_leave == TRACE_LEAVE)
#define TRACE_INPUT_OUTPUT	(1)
#define TRACE_OUTPUT_SUCCESS	(trace_leave == TRACE_LEAVE && \
				 (retcode == SQL_SUCCESS || \
				  retcode == SQL_SUCCESS_WITH_INFO))

/* Prototypes */
#include "proto.h"

#ifdef __cplusplus
}
#endif

#endif /*_IODBCDM_TRACE_H */
