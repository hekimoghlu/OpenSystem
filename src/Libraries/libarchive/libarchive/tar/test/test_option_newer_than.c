/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

DEFINE_TEST(test_option_newer_than)
{
  struct stat st;

  /*
   * Basic test of --newer-than.
   * First, create three files with different mtimes.
   * Create test1.tar with --newer-than, test2.tar without.
   */
  assertMakeDir("test1in", 0755);
  assertChdir("test1in");
  assertMakeDir("a", 0755);
  assertMakeDir("a/b", 0755);
  assertMakeFile("old.txt", 0644, "old.txt");
  assertEqualInt(0, stat("old.txt", &st));
  sleepUntilAfter(st.st_mtime);
  assertMakeFile("middle.txt", 0644, "middle.txt");
  assertEqualInt(0, stat("middle.txt", &st));
  sleepUntilAfter(st.st_mtime);
  assertMakeFile("new.txt", 0644, "new");
  assertMakeFile("a/b/new.txt", 0644, "new file in old directory");

  /* Test --newer-than on create */
  assertEqualInt(0,
	systemf("%s --format pax -cf ../test1.tar "
		"--newer-than middle.txt *.txt a", testprog));
  assertEqualInt(0,
	systemf("%s --format pax -cf ../test2.tar *.txt a", testprog));
  assertChdir("..");

  /* Extract test1.tar to a clean dir and verify what got archived. */
  assertMakeDir("test1out", 0755);
  assertChdir("test1out");
  assertEqualInt(0, systemf("%s xf ../test1.tar", testprog));
  assertFileExists("new.txt");
  assertFileExists("a/b/new.txt");
  assertFileNotExists("middle.txt");
  assertFileNotExists("old.txt");
  assertChdir("..");

  /* Extract test2.tar to a clean dir with --newer-than and verify. */
  assertMakeDir("test2out", 0755);
  assertChdir("test2out");
  assertEqualInt(0, systemf("%s xf ../test2.tar --newer-than ../test1in/middle.txt", testprog));
  assertFileExists("new.txt");
  assertFileExists("a/b/new.txt");
  assertFileNotExists("middle.txt");
  assertFileNotExists("old.txt");
  assertChdir("..");

}
