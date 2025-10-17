/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#include "wincecompat.h"

char *strError[]= {"Error 0","","No such file or directory","","","","","Arg list too long",
	"Exec format error","Bad file number","","","Not enough core","Permission denied","","",
	"","File exists","Cross-device link","","","","Invalid argument","","Too many open files",
	"","","","No space left on device","","","","","Math argument","Result too large","",
	"Resource deadlock would occur", "Unknown error under wince"};


int errno=0;

int read(int handle, char *buffer, unsigned int len)
{
	return(fread(&buffer[0], len, 1, (FILE *) handle));
}

int write(int handle, const char *buffer, unsigned int len)
{
	return(fwrite(&buffer[0], len,1,(FILE *) handle));
}

int open(const char *filename,int oflag, ...)
{
	char mode[3]; /* mode[0] ="w/r/a"  mode[1]="+" */
	mode[2]=0;
	if ( oflag==(O_WRONLY|O_CREAT) )
		mode[0]='w';
	else if (oflag==O_RDONLY)
		mode[0]='r';
	return (int) fopen(filename, mode);
}

int close(int handle)
{
	return ( fclose((FILE *) handle) );
}


char *getcwd( char *buffer, unsigned int size)
{
    /* Windows CE don't have the concept of a current directory
     * so we just return NULL to indicate an error
     */
    return NULL;
}

char *getenv( const char *varname )
{
	return NULL;
}

char *strerror(int errnum)
{
	if (errnum>MAX_STRERROR)
		return strError[MAX_STRERROR];
	else
		return strError[errnum];
}
