/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
/*
 * standalone mini error interface
 */

#ifndef _ERROR_H
#define _ERROR_H	1

#include <option.h>
#include <stdarg.h>

typedef struct Error_info_s
{
	int		errors;
	int		line;
	int		warnings;
	char*		catalog;
	char*		file;
	char*		id;
} Error_info_t;

#define ERROR_catalog(s)	s

#define ERROR_INFO	0		/* info message -- no err_id	*/
#define ERROR_WARNING	1		/* warning message		*/
#define ERROR_ERROR	2		/* error message -- no err_exit	*/
#define ERROR_FATAL	3		/* error message with err_exit	*/
#define ERROR_PANIC	ERROR_LEVEL	/* panic message with err_exit	*/

#define ERROR_LEVEL	0x00ff		/* level portion of status	*/
#define ERROR_SYSTEM	0x0100		/* report system errno message	*/
#define ERROR_USAGE	0x0800		/* usage message		*/

#define error_info	_err_info
#define error		_err_msg
#define errorv		_err_msgv

extern Error_info_t	error_info;

#define errorx(l,x,c,m)	(char*)m

extern void	error(int, ...);
extern int	errorf(void*, void*, int, ...);
extern void	errorv(const char*, int, va_list);

#endif
