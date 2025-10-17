/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
#include <fcntl.h>
#include <sys/mman.h>
#include <darwintest_utils.h>
#include <mach/vm_page_size.h>

/** Verify that F_ADDSIGS does not page fault off the end of the user blob
 * 1. Find VA space for 3 pages
 * 2. Unmap the last page
 * 3. Start fs_blob_start at PAGE_SIZE + 1 bytes away from the end of the
 * VA region (such that any read of more than PAGE_SIZE + 1 bytes will fault)
 * 4. Call fcntl with the arguments and verify the output is not EFAULT
 */
T_DECL(fcntl_addsig, "Verify that fcntl(F_ADDSIGS) doesn't EFAULT", T_META_NAMESPACE("xnu.vfs")) {
	void* blob_space = mmap(NULL, vm_page_size * 3, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	T_ASSERT_NE(blob_space, MAP_FAILED, "Blob Region: %p [%zd]", blob_space, vm_page_size);

	T_ASSERT_POSIX_SUCCESS(munmap((char*)blob_space + (vm_page_size * 2), vm_page_size), NULL);

	size_t blob_size = vm_page_size + 1;
	char* blob_start = ((char*)blob_space) + (vm_page_size * 2) - blob_size;
	fsignatures_t args = { .fs_file_start = 0, .fs_blob_start =  blob_start, .fs_blob_size = blob_size};

	// Create test file to operate on
	const char * tmp_dir = dt_tmpdir();
	char tmp_file_name[PATH_MAX];
	sprintf(tmp_file_name, "%s/foo", tmp_dir);
	FILE* tmp_file = fopen(tmp_file_name, "wx");
	fprintf(tmp_file, "Just some random content");
	fclose(tmp_file);

	int fd = open(tmp_file_name, O_RDONLY);
	T_ASSERT_POSIX_SUCCESS(fd, "tmp file: %s", tmp_file_name);

	// This command will fail, but should not fail with EFAULT
	int result = fcntl(fd, F_ADDSIGS, &args);
	int error = errno;
	T_QUIET; T_EXPECT_EQ(result, -1, NULL);
	// EBADEXEC is expected, but not required for success of this test
	T_EXPECT_NE(error, EFAULT, "fcntl: %d (%d:%s)", result, error, strerror(error));
}
