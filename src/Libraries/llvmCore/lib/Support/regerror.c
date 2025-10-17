/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include "regex_impl.h"

#include "regutils.h"

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

static const char *regatoi(const llvm_regex_t *, char *, int);

static struct rerr {
	int code;
	const char *name;
	const char *explain;
} rerrs[] = {
	{ REG_NOMATCH,	"REG_NOMATCH",	"llvm_regexec() failed to match" },
	{ REG_BADPAT,	"REG_BADPAT",	"invalid regular expression" },
	{ REG_ECOLLATE,	"REG_ECOLLATE",	"invalid collating element" },
	{ REG_ECTYPE,	"REG_ECTYPE",	"invalid character class" },
	{ REG_EESCAPE,	"REG_EESCAPE",	"trailing backslash (\\)" },
	{ REG_ESUBREG,	"REG_ESUBREG",	"invalid backreference number" },
	{ REG_EBRACK,	"REG_EBRACK",	"brackets ([ ]) not balanced" },
	{ REG_EPAREN,	"REG_EPAREN",	"parentheses not balanced" },
	{ REG_EBRACE,	"REG_EBRACE",	"braces not balanced" },
	{ REG_BADBR,	"REG_BADBR",	"invalid repetition count(s)" },
	{ REG_ERANGE,	"REG_ERANGE",	"invalid character range" },
	{ REG_ESPACE,	"REG_ESPACE",	"out of memory" },
	{ REG_BADRPT,	"REG_BADRPT",	"repetition-operator operand invalid" },
	{ REG_EMPTY,	"REG_EMPTY",	"empty (sub)expression" },
	{ REG_ASSERT,	"REG_ASSERT",	"\"can't happen\" -- you found a bug" },
	{ REG_INVARG,	"REG_INVARG",	"invalid argument to regex routine" },
	{ 0,		"",		"*** unknown regexp error code ***" }
};

/*
 - llvm_regerror - the interface to error numbers
 = extern size_t llvm_regerror(int, const llvm_regex_t *, char *, size_t);
 */
/* ARGSUSED */
size_t
llvm_regerror(int errcode, const llvm_regex_t *preg, char *errbuf, size_t errbuf_size)
{
	struct rerr *r;
	size_t len;
	int target = errcode &~ REG_ITOA;
	const char *s;
	char convbuf[50];

	if (errcode == REG_ATOI)
		s = regatoi(preg, convbuf, sizeof convbuf);
	else {
		for (r = rerrs; r->code != 0; r++)
			if (r->code == target)
				break;
	
		if (errcode&REG_ITOA) {
			if (r->code != 0) {
				assert(strlen(r->name) < sizeof(convbuf));
				(void) llvm_strlcpy(convbuf, r->name, sizeof convbuf);
			} else
				(void)snprintf(convbuf, sizeof convbuf,
				    "REG_0x%x", target);
			s = convbuf;
		} else
			s = r->explain;
	}

	len = strlen(s) + 1;
	if (errbuf_size > 0) {
		llvm_strlcpy(errbuf, s, errbuf_size);
	}

	return(len);
}

/*
 - regatoi - internal routine to implement REG_ATOI
 */
static const char *
regatoi(const llvm_regex_t *preg, char *localbuf, int localbufsize)
{
	struct rerr *r;

	for (r = rerrs; r->code != 0; r++)
		if (strcmp(r->name, preg->re_endp) == 0)
			break;
	if (r->code == 0)
		return("0");

	(void)snprintf(localbuf, localbufsize, "%d", r->code);
	return(localbuf);
}
