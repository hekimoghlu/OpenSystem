/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#include <config.h>

#include "roken.h"
#include "search.h"

struct node {
    char *string;
    int order;
};

extern void *rk_tdelete(const void *, void **,
		 int (*)(const void *, const void *));
extern void *rk_tfind(const void *, void * const *,
	       int (*)(const void *, const void *));
extern void *rk_tsearch(const void *, void **, int (*)(const void *, const void *));
extern void rk_twalk(const void *, void (*)(const void *, VISIT, int));

void *rootnode = NULL;
int numerr = 0;

/*
 *  This routine compares two nodes, based on an
 *  alphabetical ordering of the string field.
 */
int
node_compare(const void *node1, const void *node2)
{
    return strcmp(((const struct node *) node1)->string,
		  ((const struct node *) node2)->string);
}

static int walkorder = -1;

void
list_node(const void *ptr, VISIT order, int level)
{
    const struct node *p = *(const struct node **) ptr;

    if (order == postorder || order == leaf)  {
	walkorder++;
	if (p->order != walkorder) {
	    warnx("sort failed: expected %d next, got %d\n", walkorder,
		  p->order);
	    numerr++;
	}
    }
}

int
main(int argc, char **argv)
{
    int numtest = 1;
    struct node *t, *p, tests[] = {
	{ "", 0 },
	{ "ab", 3 },
	{ "abc", 4 },
	{ "abcdefg", 8 },
	{ "abcd", 5 },
	{ "a", 2 },
	{ "abcdef", 7 },
	{ "abcde", 6 },
	{ "=", 1 },
	{ NULL }
    };

    for(t = tests; t->string; t++) {
	/* Better not be there */
	p = (struct node *)rk_tfind((void *)t, (void **)&rootnode,
				    node_compare);

	if (p) {
	    warnx("erroneous list: found %d\n", p->order);
	    numerr++;
	}

	/* Put node into the tree. */
	p = (struct node *) rk_tsearch((void *)t, (void **)&rootnode,
				       node_compare);

	if (!p) {
	    warnx("erroneous list: missing %d\n", t->order);
	    numerr++;
	}
    }

    rk_twalk(rootnode, list_node);

    for(t = tests; t->string; t++) {
	/* Better be there */
	p =  (struct node *) rk_tfind((void *)t, (void **)&rootnode,
				      node_compare);

	if (!p) {
	    warnx("erroneous list: missing %d\n", t->order);
	    numerr++;
	}

	/* pull out node */
	(void) rk_tdelete((void *)t, (void **)&rootnode,
			  node_compare);

	/* Better not be there */
	p =  (struct node *) rk_tfind((void *)t, (void **)&rootnode,
				      node_compare);

	if (p) {
	    warnx("erroneous list: found %d\n", p->order);
	    numerr++;
	}

    }

    return numerr;
}
