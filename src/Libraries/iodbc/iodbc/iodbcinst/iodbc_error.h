/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#ifndef _IODBC_ERROR_H
#define _IODBC_ERROR_H

/* Definition of the error code array */
#define ERROR_NUM 8

extern DWORD ierror[ERROR_NUM];
extern LPSTR errormsg[ERROR_NUM];
extern SWORD numerrors;

#define CLEAR_ERROR() \
	numerrors = -1;

#define PUSH_ERROR(error) \
	if(numerrors < ERROR_NUM) \
	{ \
		ierror[++numerrors] = (error); \
		errormsg[numerrors] = NULL; \
	}

#define POP_ERROR(error) \
	if(numerrors != -1) \
	{ \
		errormsg[numerrors] = NULL; \
		(error) = ierror[numerrors--]; \
	}

#ifdef IS_ERROR
#  undef IS_ERROR
#endif
#define IS_ERROR() \
	(numerrors != -1) ? 1 : 0

#endif
