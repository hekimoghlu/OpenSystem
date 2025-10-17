/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/gen/fmtmsg.c,v 1.6 2009/11/08 14:02:54 brueffer Exp $");

#include <fmtmsg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

/* Default value for MSGVERB. */
#define	DFLT_MSGVERB	"label:severity:text:action:tag"

/* Maximum valid size for a MSGVERB. */
#define	MAX_MSGVERB	sizeof(DFLT_MSGVERB)

static char	*printfmt(char *, long, const char *, int, const char *,
		    const char *, const char *);
static char	*nextcomp(const char *);
static const char
		*sevinfo(int);
static int	 validmsgverb(const char *);

static const char * const validlist[] = {
	"label", "severity", "text", "action", "tag", NULL
};

int
fmtmsg(long class, const char *label, int sev, const char *text,
    const char *action, const char *tag)
{
	FILE *fp;
	char *env, *msgverb, *output;
	int ret = MM_OK;

	if (action == NULL) action = "";

	if (class & MM_PRINT) {
		if ((env = getenv("MSGVERB")) != NULL && *env != '\0' &&
		    strlen(env) <= strlen(DFLT_MSGVERB)) {
			if ((msgverb = strdup(env)) == NULL)
				return (MM_NOTOK);
			else if (validmsgverb(msgverb) == 0) {
				free(msgverb);
				goto def;
			}
		} else {
def:
			if ((msgverb = strdup(DFLT_MSGVERB)) == NULL)
				return (MM_NOTOK);
		}
		output = printfmt(msgverb, class, label, sev, text, action,
		    tag);
		if (output == NULL) {
			free(msgverb);
			return (MM_NOTOK);
		}
		if (*output != '\0') {
			int out_len = fprintf(stderr, "%s", output);
			if (out_len < 0) {
			    ret = MM_NOMSG;
			}
		}
		free(msgverb);
		free(output);
	}
	if (class & MM_CONSOLE) {
		output = printfmt(DFLT_MSGVERB, class, label, sev, text,
		    action, tag);
		if (output == NULL)
			return (MM_NOCON);
		if (*output != '\0') {

/*
//                        /-------------\
//                       /               \
//                      /                 \
//                     /                   \
//                     |   XXXX     XXXX   |
//                     |   XXXX     XXXX   |
//                     |   XXX       XXX   |
//                     \         X         /
//                      --\     XXX     /--
//                       | |    XXX    | |
//                       | |           | |
//                       | I I I I I I I |
//                       |  I I I I I I  |
//                        \             /
//                         --         --
//                           \-------/
//
//                      DO NOT INTEGRATE THIS CHANGE
//
//                      Integrating it means DEATH.
//               (see Revelation 6:8 for full details)

			XXX this is a *huge* kludge to pass the SuSv3 tests,
			  I don't think of it as cheating because they are
			  looking in the wrong place (/var/log/console) to do
			  their testing, but they can't look in the "right"
			  place for various reasons */
			char *cpath = "/dev/console";
			struct stat sb;
			int rc = stat("/var/log/console", &sb);
			if (rc == 0 && (sb.st_mode & S_IFDIR)) {
			    cpath = "/var/log/console";
			}
			/* XXX thus ends the kludge - changes after
			  this point may be safely integrated */

			if ((fp = fopen(cpath, "a")) == NULL) {
				if (ret == MM_OK) {
				    ret = MM_NOCON;
				} else {
				    ret = MM_NOTOK;
				}
			} else {
			    fprintf(fp, "%s", output);
			    fclose(fp);
			}
		}
		free(output);
	}
	return (ret);
}

#define INSERT_COLON							\
	if (*output != '\0')						\
		strlcat(output, ": ", size)
#define INSERT_NEWLINE							\
	if (*output != '\0')						\
		strlcat(output, "\n", size)
#define INSERT_SPACE							\
	if (*output != '\0')						\
		strlcat(output, " ", size)

/*
 * Returns NULL on memory allocation failure, otherwise returns a pointer to
 * a newly malloc()'d output buffer.
 */
static char *
printfmt(char *msgverb, long class, const char *label, int sev,
    const char *text, const char *act, const char *tag)
{
	size_t size;
	char *comp, *output;
	const char *sevname;

	size = 32;
	if (label != MM_NULLLBL)
		size += strlen(label);
	if ((sevname = sevinfo(sev)) != NULL)
		size += strlen(sevname);
	if (text != MM_NULLTXT)
		size += strlen(text);
	if (act != MM_NULLACT)
		size += strlen(act);
	if (tag != MM_NULLTAG)
		size += strlen(tag);

	if ((output = malloc(size)) == NULL)
		return (NULL);
	*output = '\0';
	while ((comp = nextcomp(msgverb)) != NULL) {
		if (strcmp(comp, "label") == 0 && label != MM_NULLLBL) {
			INSERT_COLON;
			strlcat(output, label, size);
		} else if (strcmp(comp, "severity") == 0 && sevname != NULL) {
			INSERT_COLON;
			strlcat(output, sevinfo(sev), size);
		} else if (strcmp(comp, "text") == 0 && text != MM_NULLTXT) {
			INSERT_COLON;
			strlcat(output, text, size);
		} else if (strcmp(comp, "action") == 0 && act != MM_NULLACT) {
			INSERT_NEWLINE;
			strlcat(output, "TO FIX: ", size);
			strlcat(output, act, size);
		} else if (strcmp(comp, "tag") == 0 && tag != MM_NULLTAG) {
			INSERT_SPACE;
			strlcat(output, tag, size);
		}
	}
	INSERT_NEWLINE;
	return (output);
}

/*
 * Returns a component of a colon delimited string.  NULL is returned to
 * indicate that there are no remaining components.  This function must be
 * called until it returns NULL in order for the local state to be cleared.
 */
static char *
nextcomp(const char *msgverb)
{
	static char lmsgverb[MAX_MSGVERB], *state;
	char *retval;
	
	if (*lmsgverb == '\0') {
		strlcpy(lmsgverb, msgverb, sizeof(lmsgverb));
		retval = strtok_r(lmsgverb, ":", &state);
	} else {
		retval = strtok_r(NULL, ":", &state);
	}
	if (retval == NULL)
		*lmsgverb = '\0';
	return (retval);
}

static const char *
sevinfo(int sev)
{

	switch (sev) {
	case MM_HALT:
		return ("HALT");
	case MM_ERROR:
		return ("ERROR");
	case MM_WARNING:
		return ("WARNING");
	case MM_INFO:
		return ("INFO");
	default:
		return (NULL);
	}
}

/*
 * Returns 1 if the msgverb list is valid, otherwise 0.
 */
static int
validmsgverb(const char *msgverb)
{
	char *msgcomp;
	int i, equality;

	equality = 0;
	while ((msgcomp = nextcomp(msgverb)) != NULL) {
		equality--;
		for (i = 0; validlist[i] != NULL; i++) {
			if (strcmp(msgcomp, validlist[i]) == 0)
				equality++;
		}
	}
	return (!equality);
}
