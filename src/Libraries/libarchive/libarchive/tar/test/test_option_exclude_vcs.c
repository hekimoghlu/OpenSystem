/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

DEFINE_TEST(test_option_exclude_vcs)
{
	assertUmask(0);
	assertMakeDir("in", 0755);
	assertChdir("in");
	assertMakeFile("file", 0644, "");
	assertMakeDir("dir", 0755);
	assertMakeDir("CVS", 0755);
	assertMakeFile("CVS/fileattr", 0644, "");
	assertMakeFile(".cvsignore", 0644, "");
	assertMakeDir("RCS", 0755);
	assertMakeFile("RCS/somefile", 0655, "");
	assertMakeDir("SCCS", 0755);
	assertMakeFile("SCCS/somefile", 0655, "");
	assertMakeDir(".svn", 0755);
	assertMakeFile(".svn/format", 0655, "");
	assertMakeDir(".git", 0755);
	assertMakeFile(".git/config", 0655, "");
	assertMakeFile(".gitignore", 0644, "");
	assertMakeFile(".gitattributes", 0644, "");
	assertMakeFile(".gitmodules", 0644, "");
	assertMakeDir(".arch-ids", 0755);
	assertMakeFile(".arch-ids/somefile", 0644, "");
	assertMakeDir("{arch}", 0755);
	assertMakeFile("{arch}/somefile", 0644, "");
	assertMakeFile("=RELEASE-ID", 0644, "");
	assertMakeFile("=meta-update", 0644, "");
	assertMakeFile("=update", 0644, "");
	assertMakeDir(".bzr", 0755);
	assertMakeDir(".bzr/checkout", 0755);
	assertMakeFile(".bzrignore", 0644, "");
	assertMakeFile(".bzrtags", 0644, "");
	assertMakeDir(".hg", 0755);
	assertMakeFile(".hg/dirstate", 0644, "");
	assertMakeFile(".hgignore", 0644, "");
	assertMakeFile(".hgtags", 0644, "");
	assertMakeDir("_darcs", 0755);
	assertMakeFile("_darcs/format", 0644, "");
	assertChdir("..");
	
	assertEqualInt(0, systemf("%s -c -C in -f included.tar .", testprog));
	assertEqualInt(0,
	    systemf("%s -c --exclude-vcs -C in -f excluded.tar .", testprog));

	/* No flags, archive with vcs files */
	assertMakeDir("vcs-noexclude", 0755);
	assertEqualInt(0, systemf("%s -x -C vcs-noexclude -f included.tar",
	    testprog));
	assertChdir("vcs-noexclude");
	assertFileExists("file");
	assertIsDir("dir", 0755);
	assertIsDir("CVS", 0755);
	assertFileExists("CVS/fileattr");
	assertFileExists(".cvsignore");
	assertIsDir("RCS", 0755);
	assertFileExists("RCS/somefile");
	assertIsDir("SCCS", 0755);
	assertFileExists("SCCS/somefile");
	assertIsDir(".svn", 0755);
	assertFileExists(".svn/format");
	assertIsDir(".git", 0755);
	assertFileExists(".git/config");
	assertFileExists(".gitignore");
	assertFileExists(".gitattributes");
	assertFileExists(".gitmodules");
	assertIsDir(".arch-ids", 0755);
	assertFileExists(".arch-ids/somefile");
	assertIsDir("{arch}", 0755);
	assertFileExists("{arch}/somefile");
	assertFileExists("=RELEASE-ID");
	assertFileExists("=meta-update");
	assertFileExists("=update");
	assertIsDir(".bzr", 0755);
	assertIsDir(".bzr/checkout", 0755);
	assertFileExists(".bzrignore");
	assertFileExists(".bzrtags");
	assertIsDir(".hg", 0755);
	assertFileExists(".hg/dirstate");
	assertFileExists(".hgignore");
	assertFileExists(".hgtags");
	assertIsDir("_darcs", 0755);
	assertFileExists("_darcs/format");
	assertChdir("..");

	/* --exclude-vcs, archive with vcs files */
	assertMakeDir("vcs-exclude", 0755);
	assertEqualInt(0,
	    systemf("%s -x --exclude-vcs -C vcs-exclude -f included.tar", testprog));
	assertChdir("vcs-exclude");
	assertFileExists("file");
	assertIsDir("dir", 0755);
	assertFileNotExists("CVS");
	assertFileNotExists("CVS/fileattr");
	assertFileNotExists(".cvsignore");
	assertFileNotExists("RCS");
	assertFileNotExists("RCS/somefile");
	assertFileNotExists("SCCS");
	assertFileNotExists("SCCS/somefile");
	assertFileNotExists(".svn");
	assertFileNotExists(".svn/format");
	assertFileNotExists(".git");
	assertFileNotExists(".git/config");
	assertFileNotExists(".gitignore");
	assertFileNotExists(".gitattributes");
	assertFileNotExists(".gitmodules");
	assertFileNotExists(".arch-ids");
	assertFileNotExists(".arch-ids/somefile");
	assertFileNotExists("{arch}");
	assertFileNotExists("{arch}/somefile");
	assertFileNotExists("=RELEASE-ID");
	assertFileNotExists("=meta-update");
	assertFileNotExists("=update");
	assertFileNotExists(".bzr");
	assertFileNotExists(".bzr/checkout");
	assertFileNotExists(".bzrignore");
	assertFileNotExists(".bzrtags");
	assertFileNotExists(".hg");
	assertFileNotExists(".hg/dirstate");
	assertFileNotExists(".hgignore");
	assertFileNotExists(".hgtags");
	assertFileNotExists("_darcs");
	assertFileNotExists("_darcs/format");
	assertChdir("..");

	/* --exclude-vcs, archive without vcs files */
	assertMakeDir("novcs-exclude", 0755);
	assertEqualInt(0,
	    systemf("%s -x --exclude-vcs -C novcs-exclude -f excluded.tar",
	    testprog));
	assertChdir("novcs-exclude");
	assertFileExists("file");
	assertIsDir("dir", 0755);
	assertFileNotExists("CVS");
	assertFileNotExists("CVS/fileattr");
	assertFileNotExists(".cvsignore");
	assertFileNotExists("RCS");
	assertFileNotExists("RCS/somefile");
	assertFileNotExists("SCCS");
	assertFileNotExists("SCCS/somefile");
	assertFileNotExists(".svn");
	assertFileNotExists(".svn/format");
	assertFileNotExists(".git");
	assertFileNotExists(".git/config");
	assertFileNotExists(".gitignore");
	assertFileNotExists(".gitattributes");
	assertFileNotExists(".gitmodules");
	assertFileNotExists(".arch-ids");
	assertFileNotExists(".arch-ids/somefile");
	assertFileNotExists("{arch}");
	assertFileNotExists("{arch}/somefile");
	assertFileNotExists("=RELEASE-ID");
	assertFileNotExists("=meta-update");
	assertFileNotExists("=update");
	assertFileNotExists(".bzr");
	assertFileNotExists(".bzr/checkout");
	assertFileNotExists(".bzrignore");
	assertFileNotExists(".bzrtags");
	assertFileNotExists(".hg");
	assertFileNotExists(".hg/dirstate");
	assertFileNotExists(".hgignore");
	assertFileNotExists(".hgtags");
	assertFileNotExists("_darcs");
	assertFileNotExists("_darcs/format");
	assertChdir("..");

	/* No flags, archive without vcs files */
	assertMakeDir("novcs-noexclude", 0755);
	assertEqualInt(0,
	    systemf("%s -x -C novcs-noexclude -f excluded.tar", testprog));
	assertChdir("novcs-noexclude");
	assertFileExists("file");
	assertIsDir("dir", 0755);
	assertFileNotExists("CVS");
	assertFileNotExists("CVS/fileattr");
	assertFileNotExists(".cvsignore");
	assertFileNotExists("RCS");
	assertFileNotExists("RCS/somefile");
	assertFileNotExists("SCCS");
	assertFileNotExists("SCCS/somefile");
	assertFileNotExists(".svn");
	assertFileNotExists(".svn/format");
	assertFileNotExists(".git");
	assertFileNotExists(".git/config");
	assertFileNotExists(".gitignore");
	assertFileNotExists(".gitattributes");
	assertFileNotExists(".gitmodules");
	assertFileNotExists(".arch-ids");
	assertFileNotExists(".arch-ids/somefile");
	assertFileNotExists("{arch}");
	assertFileNotExists("{arch}/somefile");
	assertFileNotExists("=RELEASE-ID");
	assertFileNotExists("=meta-update");
	assertFileNotExists("=update");
	assertFileNotExists(".bzr");
	assertFileNotExists(".bzr/checkout");
	assertFileNotExists(".bzrignore");
	assertFileNotExists(".bzrtags");
	assertFileNotExists(".hg");
	assertFileNotExists(".hg/dirstate");
	assertFileNotExists(".hgignore");
	assertFileNotExists(".hgtags");
	assertFileNotExists("_darcs");
	assertFileNotExists("_darcs/format");
}
