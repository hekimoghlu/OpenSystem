/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#ifndef SUDOERS_PWUTIL_H
#define SUDOERS_PWUTIL_H

#define ptr_to_item(p) ((struct cache_item *)((char *)p - offsetof(struct cache_item_##p, p)))

/*
 * Generic cache element.
 */
struct cache_item {
    unsigned int refcnt;
    unsigned int type;	/* only used for gidlist */
    char registry[16];	/* AIX-specific, empty otherwise */
    /* key */
    union {
	uid_t uid;
	gid_t gid;
	char *name;
    } k;
    /* datum */
    union {
	struct passwd *pw;
	struct group *gr;
	struct group_list *grlist;
	struct gid_list *gidlist;
    } d;
};

/*
 * Container structs to simpify size and offset calculations and guarantee
 * proper alignment of struct passwd, group, gid_list and group_list.
 */
struct cache_item_pw {
    struct cache_item cache;
    struct passwd pw;
};

struct cache_item_gr {
    struct cache_item cache;
    struct group gr;
};

struct cache_item_grlist {
    struct cache_item cache;
    struct group_list grlist;
    /* actually bigger */
};

struct cache_item_gidlist {
    struct cache_item cache;
    struct gid_list gidlist;
    /* actually bigger */
};

struct cache_item *sudo_make_gritem(gid_t gid, const char *group);
struct cache_item *sudo_make_grlist_item(const struct passwd *pw, char * const *groups);
struct cache_item *sudo_make_gidlist_item(const struct passwd *pw, char * const *gids, unsigned int type);
struct cache_item *sudo_make_pwitem(uid_t uid, const char *user);

#endif /* SUDOERS_PWUTIL_H */
