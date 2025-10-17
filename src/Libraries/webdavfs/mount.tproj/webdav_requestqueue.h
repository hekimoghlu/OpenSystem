/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
#ifndef _WEBDAV_REQUESTQUEUE_H_INCLUDE
#define _WEBDAV_REQUESTQUEUE_H_INCLUDE

#include <sys/types.h>
#include <pthread.h>
#include <mach/boolean.h>
#include <unistd.h>

#include "webdav_cache.h"
#include "webdav_network.h"

/* Functions */
#define WEBDAV_CONNECTION_UP 1
#define WEBDAV_CONNECTION_DOWN 0
extern int get_connectionstate(void);
extern void set_connectionstate(int bad);

extern int requestqueue_init(void);
extern int requestqueue_enqueue_request(int socket);
extern int requestqueue_enqueue_download(
			struct node_entry *node,			/* the node */
			struct ReadStreamRec *readStreamRecPtr); /* the ReadStreamRec */
extern int requestqueue_enqueue_server_ping(u_int32_t delay);
extern int requestqueue_purge_cache_files(void);
extern int requestqueue_enqueue_seqwrite_manager(struct stream_put_ctx *);

#endif
