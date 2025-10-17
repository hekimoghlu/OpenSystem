/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
 * FDSet.c
 * - contains FDCallout, a thin wrapper on CFSocketRef/CFFileDescriptorRef
 */
/* 
 * Modification History
 *
 * May 11, 2000		Dieter Siegmund (dieter@apple.com)
 * - created
 * June 12, 2000	Dieter Siegmund (dieter@apple.com)
 * - converted to use CFRunLoop
 * January 27, 2010	Dieter Siegmund (dieter@apple.com)
 * - use CFFileDescriptorRef for non-sockets
 */

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/errno.h>
#include <sys/socket.h>
#include <net/if_types.h>
#include <syslog.h>

#include "dynarray.h"
#include "FDSet.h"
#include "symbol_scope.h"
#include "mylog.h"
#include "IPConfigurationAgentUtil.h"

struct FDCallout {
	int			fd;
	dispatch_source_t	source;
	FDCalloutFuncRef	func;
	void *			arg1;
	void *			arg2;
};

PRIVATE_EXTERN FDCalloutRef
FDCalloutCreate(int fd, FDCalloutFuncRef func,
		void * arg1, void * arg2,
		dispatch_block_t cancel_block)
{
	FDCalloutRef		callout;
	dispatch_block_t	handler;
	struct stat		sb;

	if (fstat(fd, &sb) < 0) {
		my_log(LOG_ERR, "%s: fstat %s (%d)",
		       __func__, strerror(errno), errno);
		return (NULL);
	}
	callout = malloc(sizeof(*callout));
	bzero(callout, sizeof(*callout));
	callout->fd = fd;
	callout->func = func;
	callout->arg1 = arg1;
	callout->arg2 = arg2;
	callout->source = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ,
						 fd, 0,
						 IPConfigurationAgentQueue());
	dispatch_source_set_cancel_handler(callout->source, cancel_block);
	handler = ^{
		if (callout->func != NULL) {
			(*callout->func)(callout->arg1, callout->arg2);
		}
	};
	dispatch_source_set_event_handler(callout->source, handler);
	dispatch_activate(callout->source);
	return (callout);
}

PRIVATE_EXTERN void
FDCalloutRelease(FDCalloutRef * callout_p)
{
	FDCalloutRef callout = *callout_p;

	if (callout == NULL) {
		return;
	}
	if (callout->source) {
		dispatch_source_cancel(callout->source);
		dispatch_release(callout->source);
		callout->source = NULL;
	}
	free(callout);
	*callout_p = NULL;
	return;
}

PRIVATE_EXTERN int
FDCalloutGetFD(FDCalloutRef callout)
{
	return (callout->fd);
}
