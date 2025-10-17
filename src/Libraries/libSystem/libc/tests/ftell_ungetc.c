/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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
#include <darwintest.h>

#include <assert.h>
#include <string.h>

#include <stdio.h>

T_DECL(ftell_ungetc, "Test interactions of ftell and ungetc")
{
	FILE *fp = stdin;
	fp = fopen("assets/ftell_ungetc.txt", "rb");
	T_QUIET; T_ASSERT_NE(fp, (FILE*)NULL, "Open the file");

	/* Test ftell without having done any reads/writes */
	T_ASSERT_EQ(ftell(fp), 0L, "ftell without having done any reads/writes");

	/* Read one character */
	T_ASSERT_EQ(fgetc(fp), '/', "Read one charatcter");

	/* Check ftell again after one read */
	T_ASSERT_EQ(ftell(fp), 1L, "ftell after one read");

	/* Push back one character */
	T_ASSERT_EQ(ungetc('/', fp), '/', "push back one character");

	/* Check ftell again after pushing back one char */
	T_ASSERT_EQ(ftell(fp), 0L, "ftell after pushing back one char");

	/* Read it again */
	T_ASSERT_EQ(fgetc(fp), '/', "read pushed back character again");
	T_ASSERT_EQ(ftell(fp), 1L, "ftell after reading again");

	/* Seek and test ftell again */
	T_ASSERT_EQ(fseek(fp, 2, SEEK_SET), 0, "seek");
	T_ASSERT_EQ(ftell(fp), 2L, "ftell after seeking");

	/* Push back invalid char (EOF) */
	T_ASSERT_EQ(ungetc(EOF, fp), EOF, "push back invalid char");
	T_ASSERT_EQ(ftell(fp), 2L, "ftell after pushing invalid char, pos should not have changed");

	/* Read, push back different char, read again
	 * and check ftell.
	 *
	 * Cppreference:
	 * A successful call to ungetc on a text stream modifies
	 * the stream position indicator in unspecified manner but
	 * guarantees that after all pushed-back characters are
	 * retrieved with a read operation, the stream position
	 * indicator is equal to its value before ungetc.
	 */
	T_ASSERT_EQ(fgetc(fp), '-', "read another character");
	T_ASSERT_EQ(ftell(fp), 3L, "ftell after read");
	T_ASSERT_EQ(ungetc('A', fp), 'A', "push back a different character");
	T_ASSERT_EQ(fgetc(fp), 'A', "read back the different character");
	T_ASSERT_EQ(ftell(fp), 3L, "ftell after pushback and read back");

	/* Push back a non-read character and test ftell.
	 *
	 * According to POSIX:
	 * The file-position indicator is decremented by each
	 * successful call to ungetc();
	 *
	 * Cppreference:
	 * A successful call to ungetc on a binary stream decrements
	 * the stream position indicator by one
	 */
	T_ASSERT_EQ(fgetc(fp), '+', "read another character");
	T_ASSERT_EQ(ungetc('A', fp), 'A', "push back a different character");
	T_EXPECTFAIL; T_ASSERT_EQ(ftell(fp), 3L, "ftell after pushback - EXPECTED FAIL rdar://66131999");
}
