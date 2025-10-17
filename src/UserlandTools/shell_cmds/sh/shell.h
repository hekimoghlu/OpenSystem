/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#ifndef SHELL_H_
#define SHELL_H_

#include <inttypes.h>

/*
 * The follow should be set to reflect the type of system you have:
 *	JOBS -> 1 if you have Berkeley job control, 0 otherwise.
 *	define DEBUG=1 to compile in debugging (set global "debug" to turn on)
 *	define DEBUG=2 to compile in and turn on debugging.
 *
 * When debugging is on, debugging info will be written to ./trace and
 * a quit signal will generate a core dump.
 */


#define	JOBS 1
/* #define DEBUG 1 */

/*
 * Type of used arithmetics. SUSv3 requires us to have at least signed long.
 */
typedef intmax_t arith_t;
#define	ARITH_FORMAT_STR  "%" PRIdMAX
#define	atoarith_t(arg)  strtoimax(arg, NULL, 0)
#define	strtoarith_t(nptr, endptr, base)  strtoimax(nptr, endptr, base)
#define	ARITH_MIN INTMAX_MIN
#define	ARITH_MAX INTMAX_MAX

typedef void *pointer;

#include <sys/cdefs.h>

extern char nullstr[1];		/* null string */

#ifdef DEBUG
#define TRACE(param)  sh_trace param
#else
#define TRACE(param)
#endif

#endif /* !SHELL_H_ */
