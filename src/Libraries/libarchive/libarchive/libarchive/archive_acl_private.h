/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#ifndef ARCHIVE_ACL_PRIVATE_H_INCLUDED
#define ARCHIVE_ACL_PRIVATE_H_INCLUDED

#ifndef __LIBARCHIVE_BUILD
#error This header is only to be used internally to libarchive.
#endif

#include "archive_string.h"

struct archive_acl_entry {
	struct archive_acl_entry *next;
	int	type;			/* E.g., access or default */
	int	tag;			/* E.g., user/group/other/mask */
	int	permset;		/* r/w/x bits */
	int	id;			/* uid/gid for user/group */
	struct archive_mstring name;		/* uname/gname */
};

struct archive_acl {
	mode_t		mode;
	struct archive_acl_entry	*acl_head;
	struct archive_acl_entry	*acl_p;
	int		 acl_state;	/* See acl_next for details. */
	wchar_t		*acl_text_w;
	char		*acl_text;
	int		 acl_types;
};

void archive_acl_clear(struct archive_acl *);
void archive_acl_copy(struct archive_acl *, struct archive_acl *);
int archive_acl_count(struct archive_acl *, int);
int archive_acl_types(struct archive_acl *);
int archive_acl_reset(struct archive_acl *, int);
int archive_acl_next(struct archive *, struct archive_acl *, int,
    int *, int *, int *, int *, const char **);

int archive_acl_add_entry(struct archive_acl *, int, int, int, int, const char *);
int archive_acl_add_entry_w_len(struct archive_acl *,
    int, int, int, int, const wchar_t *, size_t);
int archive_acl_add_entry_len(struct archive_acl *,
    int, int, int, int, const char *, size_t);

wchar_t *archive_acl_to_text_w(struct archive_acl *, ssize_t *, int,
    struct archive *);
char *archive_acl_to_text_l(struct archive_acl *, ssize_t *, int,
    struct archive_string_conv *);

/*
 * ACL text parser.
 */
int archive_acl_from_text_w(struct archive_acl *, const wchar_t * /* wtext */,
    int /* type */);
int archive_acl_from_text_l(struct archive_acl *, const char * /* text */,
    int /* type */, struct archive_string_conv *);

#endif /* ARCHIVE_ENTRY_PRIVATE_H_INCLUDED */
