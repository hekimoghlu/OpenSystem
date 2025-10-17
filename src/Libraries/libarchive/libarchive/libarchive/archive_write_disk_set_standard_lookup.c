/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#include "archive_platform.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef HAVE_GRP_H
#include <grp.h>
#endif
#ifdef HAVE_PWD_H
#include <pwd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "archive.h"
#include "archive_private.h"
#include "archive_read_private.h"
#include "archive_write_disk_private.h"

struct bucket {
	char	*name;
	int	 hash;
	id_t	 id;
};

static const size_t cache_size = 127;
static unsigned int	hash(const char *);
static int64_t	lookup_gid(void *, const char *uname, int64_t);
static int64_t	lookup_uid(void *, const char *uname, int64_t);
static void	cleanup(void *);

/*
 * Installs functions that use getpwnam()/getgrnam()---along with
 * a simple cache to accelerate such lookups---into the archive_write_disk
 * object.  This is in a separate file because getpwnam()/getgrnam()
 * can pull in a LOT of library code (including NIS/LDAP functions, which
 * pull in DNS resolvers, etc).  This can easily top 500kB, which makes
 * it inappropriate for some space-constrained applications.
 *
 * Applications that are size-sensitive may want to just use the
 * real default functions (defined in archive_write_disk.c) that just
 * use the uid/gid without the lookup.  Or define your own custom functions
 * if you prefer.
 *
 * TODO: Replace these hash tables with simpler move-to-front LRU
 * lists with a bounded size (128 items?).  The hash is a bit faster,
 * but has a bad pathology in which it thrashes a single bucket.  Even
 * walking a list of 128 items is a lot faster than calling
 * getpwnam()!
 */
int
archive_write_disk_set_standard_lookup(struct archive *a)
{
	struct bucket *ucache = calloc(cache_size, sizeof(struct bucket));
	struct bucket *gcache = calloc(cache_size, sizeof(struct bucket));
	if (ucache == NULL || gcache == NULL) {
		free(ucache);
		free(gcache);
		return (ARCHIVE_FATAL);
	}
	archive_write_disk_set_group_lookup(a, gcache, lookup_gid, cleanup);
	archive_write_disk_set_user_lookup(a, ucache, lookup_uid, cleanup);
	return (ARCHIVE_OK);
}

static int64_t
lookup_gid(void *private_data, const char *gname, int64_t gid)
{
	int h;
	struct bucket *b;
	struct bucket *gcache = (struct bucket *)private_data;

	/* If no gname, just use the gid provided. */
	if (gname == NULL || *gname == '\0')
		return (gid);

	/* Try to find gname in the cache. */
	h = hash(gname);
	b = &gcache[h % cache_size ];
	if (b->name != NULL && b->hash == h && strcmp(gname, b->name) == 0)
		return ((gid_t)b->id);

	/* Free the cache slot for a new entry. */
	free(b->name);
	b->name = strdup(gname);
	/* Note: If strdup fails, that's okay; we just won't cache. */
	b->hash = h;
#if HAVE_GRP_H
#  if HAVE_GETGRNAM_R
	{
		char _buffer[128];
		size_t bufsize = 128;
		char *buffer = _buffer;
		char *allocated = NULL;
		struct group	grent, *result;
		int r;

		for (;;) {
			result = &grent; /* Old getgrnam_r ignores last arg. */
			r = getgrnam_r(gname, &grent, buffer, bufsize, &result);
			if (r == 0)
				break;
			if (r != ERANGE)
				break;
			bufsize *= 2;
			free(allocated);
			allocated = malloc(bufsize);
			if (allocated == NULL)
				break;
			buffer = allocated;
		}
		if (result != NULL)
			gid = result->gr_gid;
		free(allocated);
	}
#  else /* HAVE_GETGRNAM_R */
	{
		struct group *result;

		result = getgrnam(gname);
		if (result != NULL)
			gid = result->gr_gid;
	}
#  endif /* HAVE_GETGRNAM_R */
#elif defined(_WIN32) && !defined(__CYGWIN__)
	/* TODO: do a gname->gid lookup for Windows. */
#else
	#error No way to perform gid lookups on this platform
#endif
	b->id = (gid_t)gid;

	return (gid);
}

static int64_t
lookup_uid(void *private_data, const char *uname, int64_t uid)
{
	int h;
	struct bucket *b;
	struct bucket *ucache = (struct bucket *)private_data;

	/* If no uname, just use the uid provided. */
	if (uname == NULL || *uname == '\0')
		return (uid);

	/* Try to find uname in the cache. */
	h = hash(uname);
	b = &ucache[h % cache_size ];
	if (b->name != NULL && b->hash == h && strcmp(uname, b->name) == 0)
		return ((uid_t)b->id);

	/* Free the cache slot for a new entry. */
	free(b->name);
	b->name = strdup(uname);
	/* Note: If strdup fails, that's okay; we just won't cache. */
	b->hash = h;
#if HAVE_PWD_H
#  if HAVE_GETPWNAM_R
	{
		char _buffer[128];
		size_t bufsize = 128;
		char *buffer = _buffer;
		char *allocated = NULL;
		struct passwd	pwent, *result;
		int r;

		for (;;) {
			result = &pwent; /* Old getpwnam_r ignores last arg. */
			r = getpwnam_r(uname, &pwent, buffer, bufsize, &result);
			if (r == 0)
				break;
			if (r != ERANGE)
				break;
			bufsize *= 2;
			free(allocated);
			allocated = malloc(bufsize);
			if (allocated == NULL)
				break;
			buffer = allocated;
		}
		if (result != NULL)
			uid = result->pw_uid;
		free(allocated);
	}
#  else /* HAVE_GETPWNAM_R */
	{
		struct passwd *result;

		result = getpwnam(uname);
		if (result != NULL)
			uid = result->pw_uid;
	}
#endif	/* HAVE_GETPWNAM_R */
#elif defined(_WIN32) && !defined(__CYGWIN__)
	/* TODO: do a uname->uid lookup for Windows. */
#else
	#error No way to look up uids on this platform
#endif
	b->id = (uid_t)uid;

	return (uid);
}

static void
cleanup(void *private)
{
	size_t i;
	struct bucket *cache = (struct bucket *)private;

	for (i = 0; i < cache_size; i++)
		free(cache[i].name);
	free(cache);
}


static unsigned int
hash(const char *p)
{
	/* A 32-bit version of Peter Weinberger's (PJW) hash algorithm,
	   as used by ELF for hashing function names. */
	unsigned g, h = 0;
	while (*p != '\0') {
		h = (h << 4) + *p++;
		if ((g = h & 0xF0000000) != 0) {
			h ^= g >> 24;
			h &= 0x0FFFFFFF;
		}
	}
	return h;
}
