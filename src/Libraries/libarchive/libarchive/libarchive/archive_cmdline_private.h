/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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
#ifndef ARCHIVE_CMDLINE_PRIVATE_H
#define ARCHIVE_CMDLINE_PRIVATE_H

#ifndef __LIBARCHIVE_BUILD
#ifndef __LIBARCHIVE_TEST
#error This header is only to be used internally to libarchive.
#endif
#endif

struct archive_cmdline {
        char            *path;
        char            **argv;
        int              argc;
};

struct archive_cmdline *__archive_cmdline_allocate(void);
int __archive_cmdline_parse(struct archive_cmdline *, const char *);
int __archive_cmdline_free(struct archive_cmdline *);

#endif
