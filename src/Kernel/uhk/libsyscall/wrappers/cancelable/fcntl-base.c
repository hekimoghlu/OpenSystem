/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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
#include <stdarg.h>

int __FCNTL(int, int, void *);

/*
 * Stub function to account for the differences in the size of the third
 * argument when int and void * are different sizes. Also add pthread
 * cancelability.
 */
int
fcntl(int fd, int cmd, ...)
{
	va_list ap;
	void *arg;

	va_start(ap, cmd);
	switch (cmd) {
	case F_GETLK:
	case F_GETLKPID:
	case F_SETLK:
	case F_SETLKW:
	case F_SETLKWTIMEOUT:
	case F_OFD_GETLK:
	case F_OFD_GETLKPID:
	case F_OFD_SETLK:
	case F_OFD_SETLKW:
	case F_OFD_SETLKWTIMEOUT:
	case F_PREALLOCATE:
	case F_PUNCHHOLE:
	case F_SETSIZE:
	case F_RDADVISE:
	case F_LOG2PHYS:
	case F_LOG2PHYS_EXT:
	case F_GETPATH:
	case F_GETPATH_NOFIRMLINK:
	case F_GETPATH_MTMINFO:
	case F_GETCODEDIR:
	case F_PATHPKG_CHECK:
	case F_OPENFROM:
	case F_UNLINKFROM:
	case F_ADDSIGS:
	case F_ADDFILESIGS:
	case F_ADDFILESIGS_FOR_DYLD_SIM:
	case F_ADDFILESIGS_RETURN:
	case F_ADDFILESIGS_INFO:
	case F_ADDSIGS_MAIN_BINARY:
	case F_ADDFILESUPPL:
	case F_FINDSIGS:
	case F_TRANSCODEKEY:
	case F_TRIM_ACTIVE_FILE:
	case F_SPECULATIVE_READ:
	case F_CHECK_LV:
	case F_GETSIGSINFO:
	case F_ATTRIBUTION_TAG:
	case F_ASSERT_BG_ACCESS:
		arg = va_arg(ap, void *);
		break;
	default:
		arg = (void *)((unsigned long)va_arg(ap, int));
		break;
	}
	va_end(ap);
	return __FCNTL(fd, cmd, arg);
}
