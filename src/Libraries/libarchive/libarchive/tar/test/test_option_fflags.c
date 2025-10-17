/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#include "test.h"

#if defined(_WIN32) && !defined(__CYGWIN__) && !defined(__BORLANDC__)
#define chmod _chmod
#endif

static void
clear_fflags(const char *pathname)
{
#if defined(HAVE_STRUCT_STAT_ST_FLAGS)
	chflags(pathname, 0);
#elif (defined(FS_IOC_GETFLAGS) && defined(HAVE_WORKING_FS_IOC_GETFLAGS)) || \
      (defined(EXT2_IOC_GETFLAGS) && defined(HAVE_WORKING_EXT2_IOC_GETFLAGS))
	int fd;

	fd = open(pathname, O_RDONLY | O_NONBLOCK);
	if (fd < 0)
		return;
	ioctl(fd,
#ifdef FS_IOC_GETFLAGS
	    FS_IOC_GETFLAGS,
#else
	    EXT2_IOC_GETFLAGS,
#endif
	    0);
#else
	(void)pathname; /* UNUSED */
#endif
	return;
}

DEFINE_TEST(test_option_fflags)
{
	int r;

	if (!canNodump()) {
		skipping("Can't test nodump flag on this filesystem");
		return;
	}

	/* Create a file. */
	assertMakeFile("f", 0644, "a");

	/* Set nodump flag on the file */
	assertSetNodump("f");
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
 */chmod("f", 0644);

	/* Archive it with fflags */
	r = systemf("%s -c --fflags -f fflags.tar f >fflags.out 2>fflags.err", testprog);
	assertEqualInt(r, 0);

	/* Archive it without fflags */
	r = systemf("%s -c --no-fflags -f nofflags.tar f >nofflags.out 2>nofflags.err", testprog);
	assertEqualInt(r, 0);

	/* Extract fflags with fflags */
	assertMakeDir("fflags_fflags", 0755);
	clear_fflags("fflags_fflags");
	r = systemf("%s -x -C fflags_fflags --no-same-permissions --fflags -f fflags.tar >fflags_fflags.out 2>fflags_fflags.err", testprog);
	assertEqualInt(r, 0);
	assertEqualFflags("f", "fflags_fflags/f");

	/* Extract fflags without fflags */
	assertMakeDir("fflags_nofflags", 0755);
	clear_fflags("fflags_nofflags");
	r = systemf("%s -x -C fflags_nofflags -p --no-fflags -f fflags.tar >fflags_nofflags.out 2>fflags_nofflags.err", testprog);
	assertEqualInt(r, 0);
	assertUnequalFflags("f", "fflags_nofflags/f");

	/* Extract nofflags with fflags */
	assertMakeDir("nofflags_fflags", 0755);
	clear_fflags("nofflags_fflags");
	r = systemf("%s -x -C nofflags_fflags --no-same-permissions --fflags -f nofflags.tar >nofflags_fflags.out 2>nofflags_fflags.err", testprog);
	assertEqualInt(r, 0);	
	assertUnequalFflags("f", "nofflags_fflags/f");

	/* Extract nofflags with nofflags */
	assertMakeDir("nofflags_nofflags", 0755);
	clear_fflags("nofflags_nofflags");
	r = systemf("%s -x -C nofflags_nofflags -p --no-fflags -f nofflags.tar >nofflags_nofflags.out 2>nofflags_nofflags.err", testprog);
	assertEqualInt(r, 0);
	assertUnequalFflags("f", "nofflags_nofflags/f");
}

