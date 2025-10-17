/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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
#ifndef _PATHWATCH_H_
#define _PATHWATCH_H_

#include <dispatch/dispatch.h>
/*
 * types for virtual path nodes (path_node_t)
 */
#define PATH_NODE_TYPE_GHOST 0
#define PATH_NODE_TYPE_FILE  1
#define PATH_NODE_TYPE_LINK  2
#define PATH_NODE_TYPE_DIR   3
#define PATH_NODE_TYPE_OTHER 4


enum
{
	PATH_NODE_DELETE = 0x0001, /* node or path deleted */
	PATH_NODE_WRITE  = 0x0002, /* node written */
	PATH_NODE_EXTEND = 0x0004, /* node extended */
	PATH_NODE_ATTRIB = 0x0008, /* node attributes changed (mtime or ctime) */
	PATH_NODE_LINK	 = 0x0010, /* node link count changed */
	PATH_NODE_RENAME = 0x0020, /* node renamed, always accompanied by PATH_NODE_DELETE */
	PATH_NODE_REVOKE = 0x0040, /* access revoked, always accompanied by PATH_NODE_DELETE */
	PATH_NODE_CREATE = 0x0080, /* path created or access re-acquired */
	PATH_NODE_MTIME  = 0x0100, /* path mtime changed, always accompanied by PATH_NODE_ATTRIB */
	PATH_NODE_CTIME  = 0x0200  /* path ctime changed, always accompanied by PATH_NODE_ATTRIB */
};

/* all bits mask */
#define PATH_NODE_ALL 0x000003ff
/* src is suspended */
#define PATH_SRC_SUSPENDED 0x10000000
/* the client is notifyd */
#define PATH_NODE_CLIENT_NOTIFYD 0x20000000

/* Path changes coalesce for 100 milliseconds */
#define PNODE_COALESCE_TIME 100000000

/*
 * path_node_t represents a virtual path
 */
typedef struct
{
	char *path;
	size_t plen;
	audit_token_t audit;
	uint32_t pname_count;
	char **pname;
	uint32_t type;
	uint32_t flags;
	dispatch_source_t src;
	void *contextp;
	uint32_t context32;
	uint64_t context64;
	uint32_t refcount;
} path_node_t;

path_node_t *path_node_create(const char *path, audit_token_t audit, bool is_notifyd, uint32_t mask);
void path_node_close(path_node_t *pnode);

#endif /* _PATHWATCH_H_ */
