/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include <stdlib.h>
#include <string.h>
#include <err.h>
#include <sys/param.h>
#include <arpa/inet.h>
#include <stdio.h>

#include "locate.h"

char 	**colon(char **, char*, char*);
char 	*patprep(char *);
void print_matches(u_int);
u_char 	*tolower_word(u_char *);
int 	getwm(caddr_t);
int 	getwf(FILE *);
int	check_bigram_char(int);

/* 
 * Validate bigram chars. If the test failed the database is corrupt 
 * or the database is obviously not a locate database.
 */
int
check_bigram_char(ch)
	int ch;
{
	/* legal bigram: 0, ASCII_MIN ... ASCII_MAX */
	if (ch == 0 ||
	    (ch >= ASCII_MIN && ch <= ASCII_MAX))
		return(ch);

	errx(1,
		"locate database header corrupt, bigram char outside 0, %d-%d: %d",  
		ASCII_MIN, ASCII_MAX, ch);
}

/* split a colon separated string into a char vector
 *
 * "bla:foo" -> {"foo", "bla"}
 * "bla:"    -> {"foo", dot}
 * "bla"     -> {"bla"}
 * ""	     -> do nothing
 *
 */
char **
colon(char **dbv, char *path, char *dot)
{
	int vlen;
	long slen;
	char *c, *ch, *p;
	char **pv;

	if (dbv == NULL) {
		if ((dbv = malloc(sizeof(char *))) == NULL)
			err(1, "malloc");
		*dbv = NULL;
	}

	/* empty string */
	if (*path == '\0') {
		warnx("empty database name, ignored");
		return(dbv);
	}

	/* length of string vector */
	for(vlen = 0, pv = dbv; *pv != NULL; pv++, vlen++);

	for (ch = c = path; ; ch++) {
		if (*ch == ':' ||
		    (!*ch && !(*(ch - 1) == ':' && ch == 1+ path))) {
			/* single colon -> dot */
			if (ch == c)
				p = dot;
			else {
				/* a string */
				slen = ch - c;
				if ((p = malloc(sizeof(char) * (slen + 1))) 
				    == NULL)
					err(1, "malloc");
				bcopy(c, p, slen);
				*(p + slen) = '\0';
			}
			/* increase dbv with element p */
			if ((dbv = realloc(dbv, sizeof(char *) * (vlen + 2)))
			    == NULL)
				err(1, "realloc");
			*(dbv + vlen) = p;
			*(dbv + ++vlen) = NULL;
			c = ch + 1;
		}
		if (*ch == '\0')
			break;
	}
	return (dbv);
}

void 
print_matches(counter)
	u_int counter;
{
	(void)printf("%d\n", counter);
}


/*
 * extract last glob-free subpattern in name for fast pre-match; prepend
 * '\0' for backwards match; return end of new pattern
 */
static char globfree[100];

char *
patprep(name)
	char *name;
{
	register char *endmark, *p, *subp;

	subp = globfree;
	*subp++ = '\0';   /* set first element to '\0' */
	p = name + strlen(name) - 1;

	/* skip trailing metacharacters */
	for (; p >= name; p--)
		if (strchr(LOCATE_REG, *p) == NULL)
			break;

	/* 
	 * check if maybe we are in a character class
	 *
	 * 'foo.[ch]'
	 *        |----< p
	 */
	if (p >= name && 
	    (strchr(p, '[') != NULL || strchr(p, ']') != NULL)) {
		for (p = name; *p != '\0'; p++)
			if (*p == ']' || *p == '[')
				break;
		p--;

		/* 
		 * cannot find a non-meta character, give up
		 * '*\*[a-z]'
		 *    |-------< p
		 */
		if (p >= name && strchr(LOCATE_REG, *p) != NULL)
			p = name - 1;
	}
	
	if (p < name) 			
		/* only meta chars: "???", force '/' search */
		*subp++ = '/';

	else {
		for (endmark = p; p >= name; p--)
			if (strchr(LOCATE_REG, *p) != NULL)
				break;
		for (++p;
		    (p <= endmark) && subp < (globfree + sizeof(globfree));)
			*subp++ = *p++;
	}
	*subp = '\0';
	return(--subp);
}

/* tolower word */
u_char *
tolower_word(word)
	u_char *word;
{
	register u_char *p;

	for(p = word; *p != '\0'; p++)
		*p = TOLOWER(*p);

	return(word);
}


/*
 * Read integer from mmap pointer.
 * Essentially a simple ``return *(int *)p'' but avoids sigbus
 * for integer alignment (SunOS 4.x, 5.x).
 *
 * Convert network byte order to host byte order if necessary.
 * So we can read a locate database on FreeBSD/i386 (little endian)
 * which was built on SunOS/sparc (big endian).
 */

int
getwm(p)
	caddr_t p;
{
	union {
		char buf[INTSIZE];
		int i;
	} u;
	register int i, hi;

	for (i = 0; i < (int)INTSIZE; i++)
		u.buf[i] = *p++;

	i = u.i;

	if (i > MAXPATHLEN || i < -(MAXPATHLEN)) {
		hi = ntohl(i);
		if (hi > MAXPATHLEN || hi < -(MAXPATHLEN))
			errx(1, "integer out of +-MAXPATHLEN (%d): %u",
			    MAXPATHLEN, abs(i) < abs(hi) ? i : hi);
		return(hi);
	}
	return(i);
}

/*
 * Read integer from stream.
 *
 * Convert network byte order to host byte order if necessary.
 * So we can read on FreeBSD/i386 (little endian) a locate database
 * which was built on SunOS/sparc (big endian).
 */

int
getwf(fp)
	FILE *fp;
{
	register int word, hword;

	word = getw(fp);

	if (word > MAXPATHLEN || word < -(MAXPATHLEN)) {
		hword = ntohl(word);
		if (hword > MAXPATHLEN || hword < -(MAXPATHLEN))
			errx(1, "integer out of +-MAXPATHLEN (%d): %u",
			    MAXPATHLEN, abs(word) < abs(hword) ? word : hword);
		return(hword);
	}
	return(word);
}
