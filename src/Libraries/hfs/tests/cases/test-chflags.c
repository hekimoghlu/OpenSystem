/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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

#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)

#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/attr.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <CommonCrypto/CommonDigest.h>
#include <libkern/OSAtomic.h>
#include <pthread.h>
#include <spawn.h>
#include <MobileKeyBag/MobileKeyBag.h>

#include "hfs-tests.h"
#include "test-utils.h"
#include "systemx.h"

TEST(chflags, .run_as_root = true)

#define PROFILECTL	"/usr/local/bin/profilectl"
#define TEST_DIR	"/tmp"

int run_chflags(__unused test_ctx_t *ctx)
{
	// The root file system needs to be HFS
	struct statfs sfs;
	
	assert(statfs("/tmp", &sfs) == 0);
	if (strcmp(sfs.f_fstypename, "hfs")) {
		printf("chflags needs hfs as root file system - skipping.\n");
		return 0;
	}
	
	char *file;
	asprintf(&file, "%s/chflags-metadata-test.data", TEST_DIR);
	
	int fd;
	char filedata[4] = "Asdf";

	// Change system passcode
	assert_no_err(systemx(PROFILECTL, "changepass", "", "1234", NULL));

	// Unlock the system
	assert_no_err(systemx(PROFILECTL, "unlock", "1234", NULL));

	// Wait until the device is locked
	while (MKBGetDeviceLockState(NULL) != kMobileKeyBagDeviceIsUnlocked)
		sleep(1);

	assert_with_errno((fd = open(file,
								 O_CREAT | O_RDWR | O_TRUNC, 0666)) >= 0);

	check_io(write(fd, filedata, sizeof(filedata)), sizeof(filedata));

	// Set the file to class A
	assert_no_err(fcntl(fd, F_SETPROTECTIONCLASS, 1));

	assert_no_err(fchflags(fd, UF_IMMUTABLE));

	close(fd);

	// lock the system
	assert_no_err(systemx(PROFILECTL, "lock", NULL));

	// Wait until the device is locked
	while (MKBGetDeviceLockState(NULL) != kMobileKeyBagDeviceIsLocked)
		sleep(1);

	// should be able to remove the file.
	assert_no_err(systemx("/bin/unlink", file, NULL));
	free(file);

	// Unlock the system
	assert_no_err(systemx(PROFILECTL, "unlock", "1234", NULL));

	// Change system passcode back
	assert_no_err(systemx(PROFILECTL, "changepass", "1234", "", NULL));

	return 0;
}

#endif // (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
