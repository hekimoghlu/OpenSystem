/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/nls/msgcat.c,v 1.49 2005/02/01 16:04:55 phantom Exp $");

/*
 * We need a better way of handling errors than printing text.  I need
 * to add an error handling routine.
 */

#include "namespace.h"
#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <xlocale.h>
#include <nl_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <machine/endian.h>
#include <libkern/OSByteOrder.h>
#include "un-namespace.h"

#include "msgcat.h"
#include "setlocale.h"        /* for ENCODING_LEN */

#ifndef ntohll
#define ntohll(x) OSSwapBigToHostInt64(x)
#endif

#define _DEFAULT_NLS_PATH "/usr/share/nls/%L/%N.cat:/usr/share/nls/%N/%L:/usr/local/share/nls/%L/%N.cat:/usr/local/share/nls/%N/%L"

#define	TRUE	1
#define	FALSE	0

#define	NLERR		((nl_catd) -1)
#define NLRETERR(errc)  { errno = errc; return (NLERR); }

static nl_catd  loadCat(__const char *);
static int      loadSet(MCCatT *, MCSetT *);
static void     __nls_free_resources(MCCatT *, int);

nl_catd
catopen(__const char *name, int type)
{
	int             spcleft, saverr;
	char            path[PATH_MAX];
	char            *nlspath, *lang, *base, *cptr, *pathP, *tmpptr;
	char            *cptr1, *plang, *pter, *pcode;
	struct stat     sbuf;

	if (name == NULL || *name == '\0')
		NLRETERR(EINVAL);

	/* is it absolute path ? if yes, load immediately */
	if (strchr(name, '/') != NULL)
		return (loadCat(name));

	if (type == NL_CAT_LOCALE)
		lang = (char *)querylocale(LC_MESSAGES_MASK, NULL);
	else
		lang = getenv("LANG");

	if (lang == NULL || *lang == '\0' || strlen(lang) > ENCODING_LEN ||
	    (lang[0] == '.' &&
	     (lang[1] == '\0' || (lang[1] == '.' && lang[2] == '\0'))) ||
	    strchr(lang, '/') != NULL)
		lang = "C";

	if ((plang = cptr1 = strdup(lang)) == NULL)
		return (NLERR);
	if ((cptr = strchr(cptr1, '@')) != NULL)
		*cptr = '\0';
	pter = pcode = "";
	if ((cptr = strchr(cptr1, '_')) != NULL) {
		*cptr++ = '\0';
		pter = cptr1 = cptr;
	}
	if ((cptr = strchr(cptr1, '.')) != NULL) {
		*cptr++ = '\0';
		pcode = cptr;
	}

	if ((nlspath = getenv("NLSPATH")) == NULL || issetugid())
		nlspath = _DEFAULT_NLS_PATH;

	if ((base = cptr = strdup(nlspath)) == NULL) {
		saverr = errno;
		free(plang);
		errno = saverr;
		return (NLERR);
	}

	while ((nlspath = strsep(&cptr, ":")) != NULL) {
		pathP = path;
		if (*nlspath) {
			for (; *nlspath; ++nlspath) {
				if (*nlspath == '%') {
					switch (*(nlspath + 1)) {
					case 'l':
						tmpptr = plang;
						break;
					case 't':
						tmpptr = pter;
						break;
					case 'c':
						tmpptr = pcode;
						break;
					case 'L':
						tmpptr = lang;
						break;
					case 'N':
						tmpptr = (char *)name;
						break;
					case '%':
						++nlspath;
						/* fallthrough */
					default:
						if (pathP - path >=
						    sizeof(path) - 1)
							goto too_long;
						*(pathP++) = *nlspath;
						continue;
					}
					++nlspath;
			put_tmpptr:
					spcleft = sizeof(path) -
						  (pathP - path) - 1;
					if (strlcpy(pathP, tmpptr, spcleft) >=
					    spcleft) {
			too_long:
						free(plang);
						free(base);
						NLRETERR(ENAMETOOLONG);
					}
					pathP += strlen(tmpptr);
				} else {
					if (pathP - path >= sizeof(path) - 1)
						goto too_long;
					*(pathP++) = *nlspath;
				}
			}
			*pathP = '\0';
			if (stat(path, &sbuf) == 0) {
				free(plang);
				free(base);
				return (loadCat(path));
			}
		} else {
			tmpptr = (char *)name;
			--nlspath;
			goto put_tmpptr;
		}
	}
	free(plang);
	free(base);
	NLRETERR(ENOENT);
}

/*
 * We've got an odd situation here.  The odds are real good that the
 * number we are looking for is almost the same as the index.  We could
 * use the index, check the difference and do something intelligent, but
 * I haven't quite figured out what's intelligent.
 *
 * Here's a start.
 *	Take an id N.  If there are > N items in the list, then N cannot
 *	be more than N items from the start, since otherwise there would
 *	have to be duplicate items.  So we can safely set the top to N+1
 *	(after taking into account that ids start at 1, and arrays at 0)
 *
 *	Let's say we are at position P, and we are looking for N, but have
 *	V.  If N > V, then the furthest away that N could be is
 *	P + (N-V).  So we can safely set hi to P+(N-V)+1.  For example:
 *		We are looking for 10, but have 8
 *		8	?	?	?	?
 *			>=9	>=10	>=11
 *
 */

#define LOOKUP(PARENT, CHILD, ID, NUM, SET) {                    \
	lo = 0;                                                  \
	if (ID - 1 < NUM) {                              \
		cur = ID - 1;                                    \
		hi = ID;                                         \
	} else {                                                 \
		hi = NUM;                                \
		cur = (hi - lo) / 2;                             \
	}                                                        \
	while (TRUE) {                                           \
		CHILD = PARENT->SET + cur;                       \
		if (ntohl(CHILD->ID) == ID)                             \
			break;                                   \
		if (ntohl(CHILD->ID) < ID) {                            \
			lo = cur + 1;                            \
			if (hi > cur + (ID - ntohl(CHILD->ID)) + 1)     \
				hi = cur + (ID - ntohl(CHILD->ID)) + 1; \
			dir = 1;                                 \
		} else {                                         \
			hi = cur;                                \
			dir = -1;                                \
		}                                                \
		if (lo >= hi)                                    \
			return (NULL);                           \
		if (hi - lo == 1)                                \
			cur += dir;                              \
		else                                             \
			cur += ((hi - lo) / 2) * dir;            \
	}                                                        \
}

static MCSetT *
MCGetSet(MCCatT *cat, int setId)
{
	MCSetT  *set;
	int32_t    lo, hi, cur, dir;

	if (cat == NULL || setId <= 0)
		return (NULL);
	LOOKUP(cat, set, setId, cat->numSets, sets);
	if (set->invalid && loadSet(cat, set) <= 0)
		return (NULL);
	return (set);
}

static MCMsgT *
MCGetMsg(MCSetT *set, int msgId)
{
	MCMsgT  *msg;
	int32_t    lo, hi, cur, dir;

	if (set == NULL || set->invalid || msgId <= 0)
		return (NULL);
	LOOKUP(set, msg, msgId, ntohl(set->numMsgs), u.msgs);
	return (msg);
}

char *
catgets(nl_catd catd, int setId, int msgId, __const char *dflt)
{
	MCMsgT          *msg;
	MCCatT          *cat = (MCCatT *)catd;
	__const char    *cptr;

	if (catd == NULL || catd == NLERR)
		return ((char *)dflt);
	msg = MCGetMsg(MCGetSet(cat, setId), msgId);
	if (msg != NULL)
		cptr = msg->msg.str;
	else
		cptr = dflt;
	return ((char *)cptr);
}

int
catclose(nl_catd catd)
{
	MCCatT  *cat = (MCCatT *)catd;

	if (catd == NULL || catd == NLERR) {
		errno = EBADF;
		return (-1);
	}

	(void)fclose(cat->fp);
	__nls_free_resources(cat, cat->numSets);
	free(cat);
	return (0);
}

/*
 * Internal routines
 */

/* Note that only malloc failures are allowed to return an error */
static char     *_errowner = "Message Catalog System";

#define CORRUPT() {                                            \
	(void)fclose(cat->fp);                                 \
	(void)fprintf(stderr, "%s: corrupt file.", _errowner); \
	free(cat);                                             \
	NLRETERR(EFTYPE);                                      \
}

#define NOSPACE() {                                              \
	saverr = errno;                                          \
	(void)fclose(cat->fp);                                   \
	(void)fprintf(stderr, "%s: no more memory.", _errowner); \
	free(cat);                                               \
	errno = saverr;                                          \
	return (NLERR);                                          \
}

static void
__nls_free_resources(MCCatT *cat, int i)
{
	MCSetT  *set;
	int     j;

	for (j = 0; j < i; j++) {
		set = cat->sets + j;
		if (!set->invalid) {
			free(set->data.str);
			free(set->u.msgs);
		}
	}
	free(cat->sets);
}

static nl_catd
loadCat(__const char *catpath)
{
	MCHeaderT       header;
	MCCatT          *cat;
	MCSetT          *set;
	int32_t         i;
	off_t           nextSet;
	int             saverr;
	int		fd;

	if ((cat = (MCCatT *)malloc(sizeof(MCCatT))) == NULL)
		return (NLERR);

	if ((fd = open(catpath, O_RDONLY | O_CLOEXEC)) == -1) {
		saverr = errno;
		free(cat);
		errno = saverr;
		return (NLERR);
	}

	if ((cat->fp = fdopen(fd, "r")) == NULL) {
		saverr = errno;
		close(fd);
		free(cat);
		errno = saverr;
		return (NLERR);
	}

	if (fread(&header, sizeof(header), 1, cat->fp) != 1 ||
	    strncmp(header.magic, MCMagic, MCMagicLen) != 0)
		CORRUPT();

	if (ntohl(header.majorVer) != MCMajorVer) {
		(void)fclose(cat->fp);
		free(cat);
		if (OSSwapInt32(ntohl(header.majorVer)) == MCMajorVer) {
		    (void)fprintf(stderr, "%s: %s is the wrong byte ordering.\n", _errowner, catpath);
		} else {
		    (void)fprintf(stderr, "%s: %s is version %d, we need %d.\n", _errowner, catpath, (int)ntohl(header.majorVer), MCMajorVer);
		}
		NLRETERR(EFTYPE);
	}
	if (ntohl(header.numSets) <= 0) {
		(void)fclose(cat->fp);
		free(cat);
		(void)fprintf(stderr, "%s: %s has %d sets!\n",
		    _errowner, catpath, (int)ntohl(header.numSets));
		NLRETERR(EFTYPE);
	}

	cat->numSets = ntohl(header.numSets);
	if ((cat->sets = (MCSetT *)malloc(sizeof(MCSetT) * cat->numSets)) ==
	    NULL)
		NOSPACE();

	nextSet = ntohll(header.firstSet);
	for (i = 0; i < cat->numSets; ++i) {
		if (fseeko(cat->fp, nextSet, SEEK_SET) == -1) {
			__nls_free_resources(cat, i);
			CORRUPT();
		}

		/* read in the set header */
		set = cat->sets + i;
		if (fread(set, sizeof(*set), 1, cat->fp) != 1) {
			__nls_free_resources(cat, i);
			CORRUPT();
		}

		/* if it's invalid, skip over it (and backup 'i') */
		if (set->invalid) {
			--i;
			nextSet = ntohll(set->nextSet);
			continue;
		}
		set->invalid = TRUE;
		nextSet = ntohll(set->nextSet);
	}

	return ((nl_catd) cat);
}

static int
loadSet(MCCatT *cat, MCSetT *set)
{
	MCMsgT  *msg;
	int     i;
	int     saverr;

	/* Get the data */
	if (fseeko(cat->fp, ntohll(set->data.off), SEEK_SET) == -1)
		return (0);
	if ((set->data.str = malloc(ntohl(set->dataLen))) == NULL)
		return (-1);
	if (fread(set->data.str, ntohl(set->dataLen), 1, cat->fp) != 1) {
		saverr = errno;
		free(set->data.str);
		errno = saverr;
		return (0);
	}

	/* Get the messages */
	if (fseeko(cat->fp, ntohll(set->u.firstMsg), SEEK_SET) == -1) {
		saverr = errno;
		free(set->data.str);
		errno = saverr;
		return (0);
	}
	if ((set->u.msgs = (MCMsgT *)malloc(sizeof(MCMsgT) * ntohl(set->numMsgs))) ==
	    NULL) {
		saverr = errno;
		free(set->data.str);
		errno = saverr;
		return (-1);
	}

	for (i = 0; i < ntohl(set->numMsgs); ++i) {
		msg = set->u.msgs + i;
		if (fread(msg, sizeof(*msg), 1, cat->fp) != 1) {
			saverr = errno;
			free(set->u.msgs);
			free(set->data.str);
			errno = saverr;
			return (0);
		}
		if (msg->invalid) {
			--i;
			continue;
		}
		msg->msg.str = (char *)(set->data.str + ntohll(msg->msg.off));
	}
	set->invalid = FALSE;
	return (1);
}
