/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#include <stdlib.h>
#include <string.h>
#include <xpc/private.h>
#include <sys/stat.h>
#include <TargetConditionals.h>
#include "configuration_profile.h"

#define NOTIFY_PATH_SERVICE "com.apple.system.notify.service.path:0x87:"
#define CPROF_PATH "/Library/Managed Preferences/mobile"

char *
configuration_profile_create_notification_key(const char *ident)
{
	char *out = NULL;

	if (ident == NULL) return NULL;

	if (ident[0] == '/')
	{
		asprintf(&out, "%s%s", NOTIFY_PATH_SERVICE, ident);
		return out;
	}

#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
	if (strchr(ident + 1, '/') != NULL) return NULL;
	asprintf(&out, "%s%s/%s.plist", NOTIFY_PATH_SERVICE, CPROF_PATH, ident);
#endif

	return out;
}

xpc_object_t
configuration_profile_copy_property_list(const char *ident)
{
	char path[MAXPATHLEN];
	void *data;
	int fd;
	struct stat sb;
	xpc_object_t out = NULL;

	if (ident == NULL) return NULL;

	path[0] = '\0';
	if (ident[0] == '/')
	{
		snprintf(path, sizeof(path), "%s", ident);
	}
#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
	else
	{
		if (strchr(ident + 1, '/') != NULL) return NULL;
		snprintf(path, sizeof(path), "%s/%s.plist", CPROF_PATH, ident);
	}
#endif

	if (path[0] == '\0') return NULL;

	fd = open(path, O_RDONLY, 0);
	if (fd < 0) return NULL;

	memset(&sb, 0, sizeof(struct stat));
	if (fstat(fd, &sb) < 0)
	{
		close(fd);
		return NULL;
	}

	data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	
	if (data != MAP_FAILED)
	{
		out = xpc_create_from_plist(data, sb.st_size);
		munmap(data, sb.st_size);
	}

	close(fd);

	return out;
}
