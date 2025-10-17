/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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
#ifndef _RES_DEBUG_H_
#define _RES_DEBUG_H_

/* Define RES_DEBUG_DISABLED to build without debug support */
#ifdef RES_DEBUG_DISABLED
#   define DprintC(cond, fmt, ...) /*empty*/
#   define Dprint(fmt, ...) /*empty*/
#   define Aerror(statp, file, string, error, address, alen) /*empty*/
#   define Perror(statp, file, string, error) /*empty*/
#else
#   include <errno.h>
#   ifndef RES_DEBUG_PREFIX
#   		define RES_DEBUG_PREFIX ""
#   endif
#   define _DebugPrint(filedesc, fmt, ...)\
		do {\
			int _saved_errno = errno;\
			fprintf(filedesc, RES_DEBUG_PREFIX fmt "\n", ##__VA_ARGS__);\
			errno = _saved_errno;\
		} while(0)
#   define DprintC(cond, fmt, ...)\
		do {\
			if (cond) {\
				_DebugPrint(stdout, fmt, ##__VA_ARGS__);\
			} else {\
			}\
		} while(0)
#   define DprintQ(cond, str, query, size)\
	do {\
		DprintC(cond, str);\
		if (cond) {\
			res_pquery(statp, query, size, stdout);\
		}\
	} while(0)
#   define Dprint(fmt, ...) DprintC((statp->options & RES_DEBUG), fmt, ##__VA_ARGS__)
#endif

#endif /* _RES_DEBUG_H_ */ 
/*! \file */
