/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

#ifdef HAVE_SYS_XATTR_H
#include <sys/xattr.h>
#endif

#if defined(__APPLE__) && defined(UF_COMPRESSED) && defined(HAVE_SYS_XATTR_H)\
	&& defined(HAVE_ZLIB_H)
static int
has_xattr(const char *filename, const char *xattrname)
{
	char *nl, *nlp;
	ssize_t r;
	int existing;

	r = listxattr(filename, NULL, 0, XATTR_SHOWCOMPRESSION);
	if (r < 0)
		return (0);
	if (r == 0)
		return (0);

	assert((nl = malloc(r)) != NULL);
	if (nl == NULL)
		return (0);

	r = listxattr(filename, nl, r, XATTR_SHOWCOMPRESSION);
	if (r < 0) {
		free(nl);
		return (0);
	}

	existing = 0;
	for (nlp = nl; nlp < nl + r; nlp += strlen(nlp) + 1) {
		if (strcmp(nlp, xattrname) == 0) {
			existing = 1;
			break;
		}
	}
	free(nl);
	return (existing);
}
#endif

/*
 * Exercise HFS+ Compression.
 */
DEFINE_TEST(test_write_disk_no_hfs_compression)
{
#if !defined(__APPLE__) || !defined(UF_COMPRESSED) || !defined(HAVE_SYS_XATTR_H)\
	|| !defined(HAVE_ZLIB_H)
	skipping("MacOS-specific HFS+ Compression test");
#else
	const char *refname = "test_write_disk_no_hfs_compression.tgz";
	struct archive *ad, *a;
	struct archive_entry *ae;
	struct stat st;

	extract_reference_file(refname);

	/*
	 * Extract an archive to disk with HFS+ Compression
	 * the file was compressed.
	 */
	assert((ad = archive_write_disk_new()) != NULL);
	assertEqualIntA(ad, ARCHIVE_OK,
	    archive_write_disk_set_standard_lookup(ad));
	assertEqualIntA(ad, ARCHIVE_OK,
	    archive_write_disk_set_options(ad,
		ARCHIVE_EXTRACT_TIME |
		ARCHIVE_EXTRACT_SECURE_SYMLINKS |
		ARCHIVE_EXTRACT_SECURE_NODOTDOT));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a,
	    refname, 512 * 20));

	assertMakeDir("hfscmp", 0755);
	assertChdir("hfscmp");

	/* Extract file1. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_extract2(a, ae, ad));
	/* Extract README. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_extract2(a, ae, ad));
	/* Extract NEWS. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_extract2(a, ae, ad));
	/* Extract Makefile. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_extract2(a, ae, ad));

	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	assertEqualIntA(ad, ARCHIVE_OK, archive_write_free(ad));

	/* Test file1. */
	assertEqualInt(0, stat("file1", &st));
	assertEqualInt(0, st.st_flags & UF_COMPRESSED);
	assertFileSize("file1", 8);
	assertEqualInt(0, has_xattr("file1", "com.apple.ResourceFork"));
	assertEqualInt(0, has_xattr("file1", "com.apple.decmpfs"));

	/* Test README. */
	assertEqualInt(0, stat("README", &st));
	assertEqualInt(UF_COMPRESSED, st.st_flags & UF_COMPRESSED);
	assertFileSize("README", 6586);
	assertEqualInt(0, has_xattr("README", "com.apple.ResourceFork"));
	assertEqualInt(1, has_xattr("README", "com.apple.decmpfs"));

	/* Test NEWS. */
	assertEqualInt(0, stat("NEWS", &st));
	assertEqualInt(0, st.st_flags & UF_COMPRESSED);
	assertFileSize("NEWS", 28438);
	assertEqualInt(0, has_xattr("NEWS", "com.apple.ResourceFork"));
	assertEqualInt(0, has_xattr("NEWS", "com.apple.decmpfs"));

	/* Test Makefile. */
	assertEqualInt(0, stat("Makefile", &st));
	assertEqualInt(UF_COMPRESSED, st.st_flags & UF_COMPRESSED);
	assertFileSize("Makefile", 1238119);
	assertEqualInt(1, has_xattr("Makefile", "com.apple.ResourceFork"));
	assertEqualInt(1, has_xattr("Makefile", "com.apple.decmpfs"));

	assertChdir("..");

	/*
	 * Extract an archive to disk without HFS+ Compression.
	 */
	assert((ad = archive_write_disk_new()) != NULL);
	assertEqualIntA(ad, ARCHIVE_OK,
	    archive_write_disk_set_standard_lookup(ad));
	assertEqualIntA(ad, ARCHIVE_OK,
	    archive_write_disk_set_options(ad,
		ARCHIVE_EXTRACT_TIME |
		ARCHIVE_EXTRACT_SECURE_SYMLINKS |
		ARCHIVE_EXTRACT_SECURE_NODOTDOT |
		ARCHIVE_EXTRACT_NO_HFS_COMPRESSION));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a,
	    refname, 512 * 20));

	assertMakeDir("nocmp", 0755);
	assertChdir("nocmp");

	/* Extract file1. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_extract2(a, ae, ad));
	/* Extract README. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_extract2(a, ae, ad));
	/* Extract NEWS. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_extract2(a, ae, ad));
	/* Extract Makefile. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_extract2(a, ae, ad));

	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	assertEqualIntA(ad, ARCHIVE_OK, archive_write_free(ad));

	/* Test file1. */
	assertEqualInt(0, stat("file1", &st));
	assertEqualInt(0, st.st_flags & UF_COMPRESSED);
	assertFileSize("file1", 8);
	assertEqualInt(0, has_xattr("file1", "com.apple.ResourceFork"));
	assertEqualInt(0, has_xattr("file1", "com.apple.decmpfs"));

	/* Test README. */
	assertEqualInt(0, stat("README", &st));
	assertEqualInt(0, st.st_flags & UF_COMPRESSED);
	assertFileSize("README", 6586);
	assertEqualInt(0, has_xattr("README", "com.apple.ResourceFork"));
	assertEqualInt(0, has_xattr("README", "com.apple.decmpfs"));

	/* Test NEWS. */
	assertEqualInt(0, stat("NEWS", &st));
	assertEqualInt(0, st.st_flags & UF_COMPRESSED);
	assertFileSize("NEWS", 28438);
	assertEqualInt(0, has_xattr("NEWS", "com.apple.ResourceFork"));
	assertEqualInt(0, has_xattr("NEWS", "com.apple.decmpfs"));

	/* Test Makefile. */
	assertEqualInt(0, stat("Makefile", &st));
	assertEqualInt(0, st.st_flags & UF_COMPRESSED);
	assertFileSize("Makefile", 1238119);
	assertEqualInt(0, has_xattr("Makefile", "com.apple.ResourceFork"));
	assertEqualInt(0, has_xattr("Makefile", "com.apple.decmpfs"));

	assertChdir("..");

	assertEqualFile("hfscmp/file1", "nocmp/file1");
	assertEqualFile("hfscmp/README", "nocmp/README");
	assertEqualFile("hfscmp/NEWS", "nocmp/NEWS");
	assertEqualFile("hfscmp/Makefile", "nocmp/Makefile");
#endif
}
