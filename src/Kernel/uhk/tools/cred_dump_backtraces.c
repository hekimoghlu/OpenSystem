/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <bsm/audit.h>

/* bad!  this is replicated in kern_credential.c.  make sure they stay in sync!
 * Or better yet have commone header file?
 */
#define MAX_STACK_DEPTH 8
struct cred_backtrace {
	int                             depth;
	uint32_t                stack[MAX_STACK_DEPTH];
};
typedef struct cred_backtrace cred_backtrace;

struct cred_debug_buffer {
	int                             next_slot;
	cred_backtrace  stack_buffer[1];
};
typedef struct cred_debug_buffer cred_debug_buffer;


main( int argc, char *argv[] )
{
	int                             err, i, j;
	size_t                  len;
	char                        *my_bufferp = NULL;
	cred_debug_buffer       *bt_buffp;
	cred_backtrace          *btp;

	/* get size of buffer we will need */
	len = 0;
	err = sysctlbyname( "kern.cred_bt", NULL, &len, NULL, 0 );
	if (err != 0) {
		printf( "sysctl failed  \n" );
		printf( "\terrno %d - \"%s\" \n", errno, strerror( errno ));
		return;
	}

	/* get a buffer for our back traces */
	my_bufferp = malloc( len );
	if (my_bufferp == NULL) {
		printf( "malloc error %d - \"%s\" \n", errno, strerror( errno ));
		return;
	}
	err = sysctlbyname( "kern.cred_bt", my_bufferp, &len, NULL, 0 );
	if (err != 0) {
		printf( "sysctl 2 failed  \n" );
		printf( "\terrno %d - \"%s\" \n", errno, strerror( errno ));
		return;
	}

	bt_buffp = (cred_debug_buffer *) my_bufferp;
	btp = &bt_buffp->stack_buffer[0];

	printf("number of traces %d \n", bt_buffp->next_slot);
	for (i = 0; i < bt_buffp->next_slot; i++, btp++) {
		printf("[%d] ", i);
		for (j = 0; j < btp->depth; j++) {
			printf("%p ", btp->stack[j]);
		}
		printf("\n");
	}

	return;
}
