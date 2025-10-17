/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
/*
 * 03-Apr-2005
 * DRI: Rob Braun <bbraun@synack.net>
 */
/*
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
 * Steven Cento <cento@apple.com>
*/

#ifndef _XAR_UTIL_H_
#define _XAR_UTIL_H_

#include <stdint.h>
#include "xar.h"


uint64_t xar_ntoh64(uint64_t num);
uint32_t xar_swap32(uint32_t num);
char *xar_get_path(xar_file_t f);
off_t	xar_get_heap_offset(xar_t x);
ssize_t xar_read_fd(int fd, void * buffer, size_t nbyte);
ssize_t xar_pread_fd(int fd, void * buffer, size_t nbyte, off_t offset);
ssize_t xar_write_fd(int fd, void * buffer, size_t nbyte);
ssize_t xar_pwrite_fd( int fd, void * buffer, size_t nbyte, off_t offset );
dev_t xar_makedev(uint32_t major, uint32_t minor);
void xar_devmake(dev_t dev, uint32_t *major, uint32_t *minor);
char* xar_safe_dirname(const char* path);

// This is used to check to see if a given path escapes from
// the extraction root.
int xar_path_issane(char* path);

// Returns a string containing the name of the next path component in path_to_advance. 
// Path to advance also gets moved forward to the start of the next component in the path.
// The returned string must be released by the caller.
char* xar_path_nextcomponent(char** path_to_advance);

// Make a lower string
char* xar_lowercase_string(const char* string);

/*!
 @abstract Returns the optimal io size of the filesystem backing the
 file at the path provided.
 */
size_t xar_optimal_io_size_at_path(const char *path);

/*!
@returns 0 if the file name is safe, < 0 if the file name is not safe. If out_filename is supplied a corrected file name is returned. You must free the returned file name
 */
int xar_is_safe_filename(const char *in_filename, char** out_filename);


#endif /* _XAR_UTIL_H_ */
