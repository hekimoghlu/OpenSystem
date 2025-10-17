/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#ifndef ARCHIVE_ENTRY_LOCALE_H_INCLUDED
#define ARCHIVE_ENTRY_LOCALE_H_INCLUDED

#ifndef __LIBARCHIVE_BUILD
#error This header is only to be used internally to libarchive.
#endif

struct archive_entry;
struct archive_string_conv;

/*
 * Utility functions to set and get entry attributes by translating
 * character-set. These are designed for use in format readers and writers.
 *
 * The return code and interface of these are quite different from other
 * functions for archive_entry defined in archive_entry.h.
 * Common return code are:
 *   Return 0 if the string conversion succeeded.
 *   Return -1 if the string conversion failed.
 */

#define archive_entry_gname_l	_archive_entry_gname_l
int _archive_entry_gname_l(struct archive_entry *,
    const char **, size_t *, struct archive_string_conv *);
#define archive_entry_hardlink_l	_archive_entry_hardlink_l
int _archive_entry_hardlink_l(struct archive_entry *,
    const char **, size_t *, struct archive_string_conv *);
#define archive_entry_pathname_l	_archive_entry_pathname_l
int _archive_entry_pathname_l(struct archive_entry *,
    const char **, size_t *, struct archive_string_conv *);
#define archive_entry_symlink_l	_archive_entry_symlink_l
int _archive_entry_symlink_l(struct archive_entry *,
    const char **, size_t *, struct archive_string_conv *);
#define archive_entry_uname_l	_archive_entry_uname_l
int _archive_entry_uname_l(struct archive_entry *,
    const char **, size_t *, struct archive_string_conv *);
#define archive_entry_acl_text_l _archive_entry_acl_text_l
int _archive_entry_acl_text_l(struct archive_entry *, int,
const char **, size_t *, struct archive_string_conv *) __LA_DEPRECATED;
#define archive_entry_acl_to_text_l _archive_entry_acl_to_text_l
char *_archive_entry_acl_to_text_l(struct archive_entry *, ssize_t *, int,
    struct archive_string_conv *);
#define archive_entry_acl_from_text_l _archive_entry_acl_from_text_l
int _archive_entry_acl_from_text_l(struct archive_entry *, const char* text,
    int type, struct archive_string_conv *);
#define archive_entry_copy_gname_l	_archive_entry_copy_gname_l
int _archive_entry_copy_gname_l(struct archive_entry *,
    const char *, size_t, struct archive_string_conv *);
#define archive_entry_copy_hardlink_l	_archive_entry_copy_hardlink_l
int _archive_entry_copy_hardlink_l(struct archive_entry *,
    const char *, size_t, struct archive_string_conv *);
#define archive_entry_copy_link_l	_archive_entry_copy_link_l
int _archive_entry_copy_link_l(struct archive_entry *,
    const char *, size_t, struct archive_string_conv *);
#define archive_entry_copy_pathname_l	_archive_entry_copy_pathname_l
int _archive_entry_copy_pathname_l(struct archive_entry *,
    const char *, size_t, struct archive_string_conv *);
#define archive_entry_copy_symlink_l	_archive_entry_copy_symlink_l
int _archive_entry_copy_symlink_l(struct archive_entry *,
    const char *, size_t, struct archive_string_conv *);
#define archive_entry_copy_uname_l	_archive_entry_copy_uname_l
int _archive_entry_copy_uname_l(struct archive_entry *,
    const char *, size_t, struct archive_string_conv *);

#endif /* ARCHIVE_ENTRY_LOCALE_H_INCLUDED */
