/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#include <config.h>

#include <string.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#include <errno.h>

#include <roken.h>

#ifndef HAVE_MKSTEMP

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
mkstemp(char *template)
{
    int start, i;
    pid_t val;
    val = getpid();
    start = strlen(template) - 1;
    while(template[start] == 'X') {
	template[start] = '0' + val % 10;
	val /= 10;
	start--;
    }

    do{
	int fd;
	fd = open(template, O_RDWR | O_CREAT | O_EXCL, 0600);
	if(fd >= 0 || errno != EEXIST)
	    return fd;
	i = start + 1;
	do{
	    if(template[i] == 0)
		return -1;
	    template[i]++;
	    if(template[i] == '9' + 1)
		template[i] = 'a';
	    if(template[i] <= 'z')
		break;
	    template[i] = 'a';
	    i++;
	}while(1);
    }while(1);
}

#endif
