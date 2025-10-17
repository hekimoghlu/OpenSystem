/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

#ifndef __TEST_FILE_HELPER_H_
#define __TEST_FILE_HELPER_H_

#define MAXFILESIZE 8589934592L

char* setup_tempdir();
int cleanup_tempdir();
int test_file_create(char* path, int thread_id, int num_threads, long long length);
int test_file_read_setup(char* path, int num_threads, long long length, long long max_file_size);
int test_file_read(char* path, int thread_id, int num_threads, long long length, long long max_file_size);
int test_file_read_cleanup(char* path, int num_threads, long long length);
int test_file_write_setup(char* path, int num_threads, long long length);
int test_file_write(char* path, int thread_id, int num_threads, long long length, long long max_file_size);
int test_file_write_cleanup(char* path, int num_threads, long long length);

#endif
