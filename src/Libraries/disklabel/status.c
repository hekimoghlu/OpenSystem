/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <CoreFoundation/CoreFoundation.h>

#include "util.h"

static char *
findRealDevice(char *dev) {
	struct dirent *dp;
	DIR *dir;
	struct stat sbuf;
	ino_t mine;
	char *dName;
	char *retval = NULL;
	
	if (stat(dev, &sbuf) == -1) {
		warn("cannot stat device file %s", dev);
		return NULL;
	}
	if (sbuf.st_nlink == 1) {   // assume this is the real device
		return NULL;
	}
#define BEGINS(x, y) (!strncmp((x), (y), strlen((y))))
	if (BEGINS(dev, "/dev/rdisk") || BEGINS(dev, "rdisk") ||
		BEGINS(dev, "/dev/disk") || BEGINS(dev, "disk")) {
		return NULL;
	}
	
	if (BEGINS(dev, "/dev/")) {
		dName = dev + 4;
	} else {
		dName = dev;
	}

#undef BEGINS
	mine = sbuf.st_ino;
	
	dir = opendir("/dev");
	while ((dp = readdir(dir))) {
		char *tmp = dp->d_name;
		char dbuf[6 + strlen(tmp)];
		memcpy(dbuf, "/dev/", 6);
		memcpy(dbuf + 5, tmp, sizeof(dbuf) - 5);
		if (!strcmp(dbuf, dName))
			continue;
		tmp = strrchr(dbuf, 's');
		if (!tmp)
			continue;
		if (dp->d_fileno == mine) {
			retval = strndup(dbuf, tmp - dbuf);
			break;
		}
	}
	closedir(dir);
	return retval;
}

void
doStatus(const char *dev) {
	char *realDev = findRealDevice((char*)dev);
	printf("Device %s\n", dev);
	if (realDev) {
		printf("\tReal device is %s\n", realDev);
	} else {
		realDev = (char*)dev;
	}
	dev = realDev;
	// XXX -- need to ensure this is an AppleLabel partition
	if (IsAppleLabel(dev) != 1) {
		printf("\t* * * NOT A VALID LABEL * * *\n");
		return;
	}
	printf("Metadata size = %u\n", GetMetadataSize(dev));
	if (VerifyChecksum(dev) == 0)
		printf("Metadata checksum is good\n");
	else
		printf("\t* * * Checksum is bad * * *\n");

	return;
}
