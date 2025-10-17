/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#include <windows.h>

#ifndef TESTVERSION
#include <isc/win32os.h>
#else
#include <stdio.h>
#endif
#include <isc/print.h>

int
isc_win32os_versioncheck(unsigned int major, unsigned int minor,
			 unsigned int spmajor, unsigned int spminor)
{
	OSVERSIONINFOEX osVer;
	DWORD typeMask;
	ULONGLONG conditionMask;

	memset(&osVer, 0, sizeof(OSVERSIONINFOEX));
	osVer.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
	typeMask = 0;
	conditionMask = 0;

	/* Optimistic: likely greater */
	osVer.dwMajorVersion = major;
	typeMask |= VER_MAJORVERSION;
	conditionMask = VerSetConditionMask(conditionMask,
					    VER_MAJORVERSION,
					    VER_GREATER);
	osVer.dwMinorVersion = minor;
	typeMask |= VER_MINORVERSION;
	conditionMask = VerSetConditionMask(conditionMask,
					    VER_MINORVERSION,
					    VER_GREATER);
	osVer.wServicePackMajor = spmajor;
	typeMask |= VER_SERVICEPACKMAJOR;
	conditionMask = VerSetConditionMask(conditionMask,
					    VER_SERVICEPACKMAJOR,
					    VER_GREATER);
	osVer.wServicePackMinor = spminor;
	typeMask |= VER_SERVICEPACKMINOR;
	conditionMask = VerSetConditionMask(conditionMask,
					    VER_SERVICEPACKMINOR,
					    VER_GREATER);
	if (VerifyVersionInfo(&osVer, typeMask, conditionMask))
		return (1);

	/* Failed: retry with equal */
	conditionMask = 0;
	conditionMask = VerSetConditionMask(conditionMask,
					    VER_MAJORVERSION,
					    VER_EQUAL);
	conditionMask = VerSetConditionMask(conditionMask,
					    VER_MINORVERSION,
					    VER_EQUAL);
	conditionMask = VerSetConditionMask(conditionMask,
					    VER_SERVICEPACKMAJOR,
					    VER_EQUAL);
	conditionMask = VerSetConditionMask(conditionMask,
					    VER_SERVICEPACKMINOR,
					    VER_EQUAL);
	if (VerifyVersionInfo(&osVer, typeMask, conditionMask))
		return (0);
	else
		return (-1);
}

#ifdef TESTVERSION
int
main(int argc, char **argv) {
	unsigned int major = 0;
	unsigned int minor = 0;
	unsigned int spmajor = 0;
	unsigned int spminor = 0;
	int ret;

	if (argc > 1) {
		--argc;
		++argv;
		major = (unsigned int) atoi(argv[0]);
	}
	if (argc > 1) {
		--argc;
		++argv;
		minor = (unsigned int) atoi(argv[0]);
	}
	if (argc > 1) {
		--argc;
		++argv;
		spmajor = (unsigned int) atoi(argv[0]);
	}
	if (argc > 1) {
		--argc;
		++argv;
		spminor = (unsigned int) atoi(argv[0]);
	}

	ret = isc_win32os_versioncheck(major, minor, spmajor, spminor);

	printf("%s major %u minor %u SP major %u SP minor %u\n",
	       ret > 0 ? "greater" : (ret == 0 ? "equal" : "less"),
	       major, minor, spmajor, spminor);
	return (ret);
}
#endif
