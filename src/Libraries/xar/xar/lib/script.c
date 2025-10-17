/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <limits.h>

#include "config.h"
#ifndef HAVE_ASPRINTF
#include "asprintf.h"
#endif
#include "xar.h"
#include "filetree.h"
#include "arcmod.h"

struct _script_context{
	int initted;
};

#define SCRIPT_CONTEXT(x) ((struct _script_context*)(*x))

int32_t xar_script_in(xar_t x, xar_file_t f, xar_prop_t p, void **in, size_t *inlen, void **context) {
	char *buf = *in;
	xar_prop_t tmpp;

	if(!SCRIPT_CONTEXT(context)){
		*context = calloc(1,sizeof(struct _script_context));
	}

	if( SCRIPT_CONTEXT(context)->initted )
		return 0;

	if( !xar_check_prop(x, "contents") )
		return 0;

	/* Sanity check *inlen, which really shouldn't be more than a 
	 * few kbytes...
	 */
	if( *inlen > INT_MAX )
		return 0;

	/*We only run on the begining of the file, so once we init, we don't run again*/
	SCRIPT_CONTEXT(context)->initted = 1;
	
	if( (*inlen > 2) && (buf[0] == '#') && (buf[1] == '!') ) {
		char *exe;
		int i;

		exe = malloc(*inlen);
		if( !exe )
			return -1;
		memset(exe, 0, *inlen);
		
		for(i = 2; (i < *inlen) && (buf[i] != '\0') && (buf[i] != '\n') && (buf[i] != ' '); ++i) {
			exe[i-2] = buf[i];
		}

		tmpp = xar_prop_pset(f, p, "contents", NULL);
		if( tmpp ) {
			xar_prop_pset(f, tmpp, "type", "script");
			xar_prop_pset(f, tmpp, "interpreter", exe);
		}
		free(exe);
	}
	return 0;
}

int32_t xar_script_done(xar_t x, xar_file_t f, xar_prop_t p, void **context) {

	if(!SCRIPT_CONTEXT(context)){
		return 0;
	}
		
	if( *context ){
		free(*context);
		*context = NULL;
	}

	return 0;
}
