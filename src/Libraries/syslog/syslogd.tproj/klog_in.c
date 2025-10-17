/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#include <TargetConditionals.h>

#if TARGET_OS_SIMULATOR
struct _not_empty;
#else

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include "daemon.h"

#define forever for(;;)

#define MY_ID "klog_in"
#define BUFF_SIZE 4096

static char inbuf[BUFF_SIZE];
static int bx;
static int kfd = -1;
static dispatch_source_t in_src;
static dispatch_queue_t in_queue;

void
klog_in_acceptdata(int fd)
{
	ssize_t len;
	uint32_t i;
	char *p, *q;
	asl_msg_t *m;

	len = read(fd, inbuf + bx, BUFF_SIZE - bx);
	if (len <= 0) return;

	p = inbuf;
	q = p + bx;

	for (i = 0; i < len; i++, q++)
	{
		if (*q == '\n')
		{
			*q = '\0';
			m = asl_input_parse(p, q - p, NULL, SOURCE_KERN);
			process_message(m, SOURCE_KERN);
			p = q + 1;
		}
	}

	if (p != inbuf)
	{
		memmove(inbuf, p, BUFF_SIZE - bx - 1);
		bx = q - p;
	}
}

int
klog_in_init()
{
	static dispatch_once_t once;

	dispatch_once(&once, ^{
		in_queue = dispatch_queue_create(MY_ID, NULL);
	});

	asldebug("%s: init\n", MY_ID);
	if (kfd >= 0) return 0;

	kfd = open(_PATH_KLOG, O_RDONLY, 0);
	if (kfd < 0)
	{
		asldebug("%s: couldn't open %s: %s\n", MY_ID, _PATH_KLOG, strerror(errno));
		return -1;
	}

	if (fcntl(kfd, F_SETFL, O_NONBLOCK) < 0)
	{
		close(kfd);
		kfd = -1;
		asldebug("%s: couldn't set O_NONBLOCK for fd %d (%s): %s\n", MY_ID, kfd, _PATH_KLOG, strerror(errno));
		return -1;
	}

	in_src = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, (uintptr_t)kfd, 0, in_queue);
	dispatch_source_set_event_handler(in_src, ^{ klog_in_acceptdata(kfd); });

	dispatch_resume(in_src);
	return 0;
}

int
klog_in_close(void)
{
	if (kfd < 0) return 1;

	dispatch_source_cancel(in_src);
	dispatch_release(in_src);
	in_src = NULL;

	close(kfd);
	kfd = -1;

	return 0;
}

int
klog_in_reset(void)
{
	return 0;
}

#endif /* !TARGET_OS_SIMULATOR */
