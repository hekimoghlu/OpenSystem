/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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

#import <Foundation/Foundation.h>
#import <IOKit/kext/OSKext.h>
#import <IOKit/kext/OSKextPrivate.h>
#import <Bom/Bom.h>
#import <APFS/APFS.h>

#import <libproc.h>
#import <stdio.h>
#import <stdbool.h>
#import <sysexits.h>
#import <unistd.h>

#import "bootcaches.h"
#import "kext_tools_util.h"
#import "kc_staging.h"

int main(int argc, char **argv)
{
	int result = EX_SOFTWARE;
	char pathbuf[PROC_PIDPATHINFO_MAXSIZE];
	struct statfs sfs = {};

	if (argc != 1) {
		LOG_ERROR("kcditto installs previously built kext collections onto the Preboot volume.");
		LOG_ERROR("It takes no arguments.");
		result = EX_USAGE;
		goto finish;
	}

	int ret = proc_pidpath(getpid(), pathbuf, sizeof(pathbuf));
	if (ret <= 0) {
		LOG_ERROR("Can't get executable path for (%d)%s: %s",
		          getpid(), argv[0], strerror(errno));
		goto finish;
	}
	if (statfs(pathbuf, &sfs) < 0) {
		goto finish;
	}

	LOG("Copying deferred prelinked kernels in %s...", sfs.f_mntonname);
	result = copyDeferredPrelinkedKernels(sfs.f_mntonname);
	if (result != EX_OK) {
		LOG_ERROR("Error copying deferred prelinked kernels (standalone)...");
	}

	LOG("Copying KCs in %s...", sfs.f_mntonname);
	result = copyKCsInVolume(sfs.f_mntonname);
	if (result != EX_OK) {
		LOG_ERROR("Error copying KCs (standalone)...");
		goto finish;
	}

finish:
	return result;
}
