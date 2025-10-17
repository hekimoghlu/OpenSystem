/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
/*
 * LC_MESSAGES database generation routines for localedef.
 */
#include <sys/cdefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include "localedef.h"
#include "parser.h"
#include "lmessages.h"

static struct lc_messages_T msgs;

void
init_messages(void)
{
	(void) memset(&msgs, 0, sizeof (msgs));
}

void
add_message(wchar_t *wcs)
{
	char *str;

	if ((str = to_mb_string(wcs)) == NULL) {
		INTERR;
		return;
	}
	free(wcs);

	switch (last_kw) {
	case T_YESSTR:
		msgs.yesstr = str;
		break;
	case T_NOSTR:
		msgs.nostr = str;
		break;
	case T_YESEXPR:
		msgs.yesexpr = str;
		break;
	case T_NOEXPR:
		msgs.noexpr = str;
		break;
	default:
		free(str);
		INTERR;
		break;
	}
}

void
dump_messages(void)
{
	FILE *f;
	char *ptr;

	if (msgs.yesstr == NULL) {
#ifndef __APPLE__
		warn("missing field 'yesstr'");
#endif
		msgs.yesstr = "";
	}
	if (msgs.nostr == NULL) {
#ifndef __APPLE__
		warn("missing field 'nostr'");
#endif
		msgs.nostr = "";
	}

	/*
	 * CLDR likes to add : separated lists for yesstr and nostr.
	 * Legacy Solaris code does not seem to grok this.  Fix it.
	 */
	if ((ptr = strchr(msgs.yesstr, ':')) != NULL)
		*ptr = 0;
	if ((ptr = strchr(msgs.nostr, ':')) != NULL)
		*ptr = 0;

	if ((f = open_category()) == NULL) {
		return;
	}

	if ((putl_category(msgs.yesexpr, f) == EOF) ||
	    (putl_category(msgs.noexpr, f) == EOF) ||
	    (putl_category(msgs.yesstr, f) == EOF) ||
	    (putl_category(msgs.nostr, f) == EOF)) {
		return;
	}
	close_category(f);
}
