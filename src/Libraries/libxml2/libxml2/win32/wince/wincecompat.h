/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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
#ifndef __WINCECOMPAT_H__
#define __WINCECOMPAT_H__

#include <stdio.h>
#include <winbase.h>

#define MAX_STRERROR 31

#define O_RDONLY       0x0000  /* open for reading only */
#define O_WRONLY       0x0001  /* open for writing only */
#define O_RDWR         0x0002  /* open for reading and writing */
#define O_APPEND       0x0008  /* writes done at eof */

#define O_CREAT        0x0100  /* create and open file */
#define O_TRUNC        0x0200  /* open and truncate */
#define O_EXCL         0x0400  /* open only if file doesn't already exist */

#define BUFSIZ 4096

extern int errno;
/* 
	Prototypes 
*/
int read(int handle, char *buffer, unsigned int len);
int write(int handle, const char *buffer, unsigned int len);
int open(const char *filename,int oflag, ...);
int close(int handle);
char *getenv( const char *varname );
char *getcwd( char *buffer, unsigned int size);
char *strerror(int errnum);

/*
	Macro'ed nonexistent function names

*/
#define snprintf _snprintf
#define vsnprintf(b,c,f,a) _vsnprintf(b,c,f,a)
#define perror(_t) MessageBox(NULL, _T("_t"), _T("Error/Warning"), MB_OK)

#endif
