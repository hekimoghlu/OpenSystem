/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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

#ifndef RUBY_WIN32_FILE_H
#define RUBY_WIN32_FILE_H

#define MAX_REPARSE_PATH_LEN 4092

enum {
    MINIMUM_REPARSE_BUFFER_PATH_LEN = 4
};
/* License: Ruby's */
typedef struct {
    ULONG  ReparseTag;
    USHORT ReparseDataLength;
    USHORT Reserved;
    union {
	struct {
	    USHORT SubstituteNameOffset;
	    USHORT SubstituteNameLength;
	    USHORT PrintNameOffset;
	    USHORT PrintNameLength;
	    ULONG  Flags;
	    WCHAR  PathBuffer[4];
	} SymbolicLinkReparseBuffer;
	struct {
	    USHORT SubstituteNameOffset;
	    USHORT SubstituteNameLength;
	    USHORT PrintNameOffset;
	    USHORT PrintNameLength;
	    WCHAR  PathBuffer[4];
	} MountPointReparseBuffer;
    };
} rb_w32_reparse_buffer_t;

#define rb_w32_reparse_buffer_size(n) \
    (sizeof(rb_w32_reparse_buffer_t) + \
     sizeof(WCHAR)*((n)-MINIMUM_REPARSE_BUFFER_PATH_LEN))

int rb_w32_read_reparse_point(const WCHAR *path, rb_w32_reparse_buffer_t *rp,
			      size_t bufsize, WCHAR **result, DWORD *len);

int lchown(const char *path, int owner, int group);
int rb_w32_ulchown(const char *path, int owner, int group);
int fchmod(int fd, int mode);
#define HAVE_FCHMOD 0

UINT rb_w32_filecp(void);
WCHAR *rb_w32_home_dir(void);

#endif	/* RUBY_WIN32_FILE_H */
