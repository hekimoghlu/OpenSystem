/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

#include "pmap_wakeup.h"

int pmap_wakeup(void)
{
	struct sockaddr_un sun;
	int fd;
	char b;

	memset(&sun, 0, sizeof(sun));

	sun.sun_family = AF_UNIX;
	strcpy(sun.sun_path, "/var/run/portmap.socket");

	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd == -1)
		return -1;

	if (connect(fd, (struct sockaddr *)&sun, sizeof(sun)) == -1) {
		close(fd);
		return -1;
	}

	read(fd, &b, sizeof(b));
	close(fd);
	return 0;
}
