/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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
#include "gen_locl.h"
#include "lex.h"

static Hashtab *htab;

static int
cmp(void *a, void *b)
{
    Symbol *s1 = (Symbol *) a;
    Symbol *s2 = (Symbol *) b;

    return strcmp(s1->name, s2->name);
}

static unsigned
hash(void *a)
{
    Symbol *s = (Symbol *) a;

    return hashjpw(s->name);
}

void
initsym(void)
{
    htab = hashtabnew(101, cmp, hash);
}


void
output_name(char *s)
{
    char *p;

    for (p = s; *p; ++p)
	if (*p == '-' || *p == '.')
	    *p = '_';
}

Symbol *
addsym(char *name)
{
    Symbol key, *s;

    key.name = name;
    s = (Symbol *) hashtabsearch(htab, (void *) &key);
    if (s == NULL) {
	s = (Symbol *) ecalloc(1, sizeof(*s));
	s->name = name;
	s->gen_name = estrdup(name);
	output_name(s->gen_name);
	s->stype = SUndefined;
	s->flags.used = 0;
	s->flags.external = 0;
	hashtabadd(htab, s);
    }
    return s;
}

static int
checkfunc(void *ptr, void *arg)
{
    Symbol *s = ptr;
    if (s->stype == SUndefined) {
	lex_error_message("%s is still undefined\n", s->name);
	*(int *) arg = 1;
    }

    if (!s->flags.used) {
	if (s->flags.external) {
	    lex_error_message("external symbol %s is no used\n", s->name);
	    *(int *) arg = 1;
	} else if (s->stype == Stype && !is_export(s->name)) {
	    lex_error_message("%s is still unused (not referenced internally or exported)\n", s->name);
	    *(int *) arg = 1;
	}
    }

    return 0;
}

int
checksymbols(void)
{
    int f = 0;
    hashtabforeach(htab, checkfunc, &f);
    return f;
}
