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
#ifndef _READLINE_H_
#define _READLINE_H_  1

#include <CoreFoundation/CFData.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Inspects a file's existence and size.  Returns a file handle or -1 on failure */
extern int inspect_file_and_size(const char* name, off_t *out_off_end);

/* Read a line from stdin into buffer as a null terminated string.  If buffer is
   non NULL use at most buffer_size bytes and return a pointer to buffer.  Otherwise
   return a newly malloced buffer.
   if EOF is read this function returns NULL.  */
extern char *readline(char *buffer, int buffer_size);

/* Read the file name into buffer.  On return outData.Data contains a newly
   malloced buffer of outData.Length bytes. Return 0 on success and -1 on failure.  */
extern int read_file(const char *name, uint8_t **outData, size_t *outLength);

extern CFDataRef copyFileContents(const char *path);

extern bool writeFileContents(const char *path, CFDataRef data);

#ifdef __cplusplus
}
#endif

#endif /* _READLINE_H_ */
