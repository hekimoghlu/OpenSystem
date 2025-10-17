/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#include <ctype.h>
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "calendar.h"

const char *fdays[] = {
	"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
	"Saturday", NULL,
};

const char *days[] = {
	"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", NULL,
};

const char *fmonths[] = {
	"January", "February", "March", "April", "May", "June", "Juli",
	"August", "September", "October", "November", "December", NULL,
};

const char *months[] = {
	"Jan", "Feb", "Mar", "Apr", "May", "Jun",
	"Jul", "Aug", "Sep", "Oct", "Nov", "Dec", NULL,
};

const char *sequences[] = {
	"First", "Second", "Third", "Fourth", "Fifth", "Last"
};

struct fixs fndays[8];		/* full national days names */
struct fixs ndays[8];		/* short national days names */
struct fixs fnmonths[13];	/* full national months names */
struct fixs nmonths[13];	/* short national month names */
struct fixs nsequences[10];	/* national sequence names */


void
setnnames(void)
{
	char buf[80];
	int i, l;
	struct tm tm;

	memset(&tm, 0, sizeof(struct tm));
	for (i = 0; i < 7; i++) {
		tm.tm_wday = i;
		strftime(buf, sizeof(buf), "%a", &tm);
		for (l = strlen(buf);
		     l > 0 && isspace((unsigned char)buf[l - 1]);
		     l--)
			;
		buf[l] = '\0';
		if (ndays[i].name != NULL)
			free(ndays[i].name);
		if ((ndays[i].name = strdup(buf)) == NULL)
			errx(1, "cannot allocate memory");
		ndays[i].len = strlen(buf);

		strftime(buf, sizeof(buf), "%A", &tm);
		for (l = strlen(buf);
		     l > 0 && isspace((unsigned char)buf[l - 1]);
		     l--)
			;
		buf[l] = '\0';
		if (fndays[i].name != NULL)
			free(fndays[i].name);
		if ((fndays[i].name = strdup(buf)) == NULL)
			errx(1, "cannot allocate memory");
		fndays[i].len = strlen(buf);
	}

	memset(&tm, 0, sizeof(struct tm));
	for (i = 0; i < 12; i++) {
		tm.tm_mon = i;
		strftime(buf, sizeof(buf), "%b", &tm);
		for (l = strlen(buf);
		     l > 0 && isspace((unsigned char)buf[l - 1]);
		     l--)
			;
		buf[l] = '\0';
		if (nmonths[i].name != NULL)
			free(nmonths[i].name);
		if ((nmonths[i].name = strdup(buf)) == NULL)
			errx(1, "cannot allocate memory");
		nmonths[i].len = strlen(buf);

		strftime(buf, sizeof(buf), "%B", &tm);
		for (l = strlen(buf);
		     l > 0 && isspace((unsigned char)buf[l - 1]);
		     l--)
			;
		buf[l] = '\0';
		if (fnmonths[i].name != NULL)
			free(fnmonths[i].name);
		if ((fnmonths[i].name = strdup(buf)) == NULL)
			errx(1, "cannot allocate memory");
		fnmonths[i].len = strlen(buf);
	}
}

void
setnsequences(char *seq)
{
	int i;
	char *p;

	p = seq;
	for (i = 0; i < 5; i++) {
		nsequences[i].name = p;
		if ((p = strchr(p, ' ')) == NULL) {
			/* Oh oh there is something wrong. Erase! Erase! */
			for (i = 0; i < 5; i++) {
				nsequences[i].name = NULL;
				nsequences[i].len = 0;
			}
			return;
		}
		*p = '\0';
		p++;
	}
	nsequences[i].name = p;

	for (i = 0; i < 5; i++) {
		nsequences[i].name = strdup(nsequences[i].name);
		nsequences[i].len = nsequences[i + 1].name - nsequences[i].name;
	}
	nsequences[i].name = strdup(nsequences[i].name);
	nsequences[i].len = strlen(nsequences[i].name);

	return;
}
