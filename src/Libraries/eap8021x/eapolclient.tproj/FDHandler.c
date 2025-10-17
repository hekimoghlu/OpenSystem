/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
 * FDHandler.c
 * - hides the details of how to get a callback when a file descriptor
 *   has data available
 * - wraps the CFSocket run-loop source
 */

/* 
 * Modification History
 *
 * October 26, 2001	Dieter Siegmund (dieter@apple.com)
 * - created (based on bootp/IPConfiguration.tproj/FDSet.c)
 */

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/errno.h>
#include <sys/socket.h>
#include <net/if_types.h>

#include "FDHandler.h"

#include <CoreFoundation/CFRunLoop.h>
#include <CoreFoundation/CFSocket.h>

struct FDHandler_s {
    CFRunLoopSourceRef	rls;
    CFSocketRef		socket;
    int			fd;
    FDHandler_func *	func;
    void *		arg1;
    void *		arg2;
};

static void
FDHandler_callback(CFSocketRef s, CFSocketCallBackType type, 
		   CFDataRef address, const void * data, void * info)
{
    FDHandler * 	handler = (FDHandler *)info;

    if (handler->func) {
	(*handler->func)(handler->arg1, handler->arg2);
    }
    return;
}

FDHandler *
FDHandler_create(int fd)
{
    CFSocketContext	context = { 0, NULL, NULL, NULL, NULL };
    FDHandler *	handler;

    handler = malloc(sizeof(*handler));
    if (handler == NULL)
	return (NULL);
    bzero(handler, sizeof(*handler));

    context.info = handler;
    handler->fd = fd;
    handler->socket 
	= CFSocketCreateWithNative(NULL, fd, kCFSocketReadCallBack,
				   FDHandler_callback, &context);
    handler->rls = CFSocketCreateRunLoopSource(NULL, handler->socket, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), handler->rls, 
		       kCFRunLoopDefaultMode);
    return (handler);
}

void
FDHandler_free(FDHandler * * handler_p)
{
    FDHandler * handler;

    if (handler_p == NULL) {
	return;
    }
    handler = *handler_p;
    if (handler) {
	if (handler->rls) {
	    /* cancel further handlers */
	    CFRunLoopRemoveSource(CFRunLoopGetCurrent(), handler->rls, 
				  kCFRunLoopDefaultMode);

	    /* remove one socket reference, close the file descriptor */
	    CFSocketInvalidate(handler->socket);

	    /* release the socket */
	    CFRelease(handler->socket);
	    handler->socket = NULL;

	    /* release the run loop source */
	    CFRelease(handler->rls);
	    handler->rls = NULL;
	}
	free(handler);
    }
    *handler_p = NULL;
    return;
}

void
FDHandler_enable(FDHandler * handler, FDHandler_func * func, 
		 void * arg1, void * arg2)
{
    handler->func = func;
    handler->arg1 = arg1;
    handler->arg2 = arg2;
    return;
}

void
FDHandler_disable(FDHandler * handler)
{
    handler->func = NULL;
    handler->arg1 = NULL;
    handler->arg2 = NULL;
    return;
}


int
FDHandler_fd(FDHandler * handler)
{
    return (handler->fd);
}
