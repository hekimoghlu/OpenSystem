/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_fatal.h"
#include "sudo_queue.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Note: HLTQ_ENTRY is intentionally in the middle of the struct
 *       to catch bad assumptions in the PREV/NEXT macros.
 */
struct test_data {
    int a;
    HLTQ_ENTRY(test_data) entries;
    char b;
};

TAILQ_HEAD(test_data_list, test_data);

/*
 * Simple tests for headless tail queue macros.
 */
int
main(int argc, char *argv[])
{
    struct test_data d1, d2, d3;
    struct test_data *hltq;
    struct test_data_list tq;
    int ch, errors = 0, ntests = 0;

    initprogname(argc > 0 ? argv[0] : "hltq_test");

    while ((ch = getopt(argc, argv, "v")) != -1) {
	switch (ch) {
	case 'v':
	    /* ignore */
	    break;
	default:
	    fprintf(stderr, "usage: %s [-v]\n", getprogname());
	    return EXIT_FAILURE;
	}
    }
    argc -= optind;
    argv += optind;

    /*
     * Initialize three data elements and concatenate them in order.
     */
    HLTQ_INIT(&d1, entries);
    d1.a = 1;
    d1.b = 'a';
    if (HLTQ_FIRST(&d1) != &d1) {
	sudo_warnx_nodebug("FAIL: HLTQ_FIRST(1 entry) doesn't return first element: got %p, expected %p", HLTQ_FIRST(&d1), &d1);
	errors++;
    }
    ntests++;
    if (HLTQ_LAST(&d1, test_data, entries) != &d1) {
	sudo_warnx_nodebug("FAIL: HLTQ_LAST(1 entry) doesn't return first element: got %p, expected %p", HLTQ_LAST(&d1, test_data, entries), &d1);
	errors++;
    }
    ntests++;
    if (HLTQ_PREV(&d1, test_data, entries) != NULL) {
	sudo_warnx_nodebug("FAIL: HLTQ_PREV(1 entry) doesn't return NULL: got %p", HLTQ_PREV(&d1, test_data, entries));
	errors++;
    }
    ntests++;

    HLTQ_INIT(&d2, entries);
    d2.a = 2;
    d2.b = 'b';

    HLTQ_INIT(&d3, entries);
    d3.a = 3;
    d3.b = 'c';

    HLTQ_CONCAT(&d1, &d2, entries);
    HLTQ_CONCAT(&d1, &d3, entries);
    hltq = &d1;

    /*
     * Verify that HLTQ_FIRST, HLTQ_LAST, HLTQ_NEXT, HLTQ_PREV
     * work as expected.
     */
    if (HLTQ_FIRST(hltq) != &d1) {
	sudo_warnx_nodebug("FAIL: HLTQ_FIRST(3 entries) doesn't return first element: got %p, expected %p", HLTQ_FIRST(hltq), &d1);
	errors++;
    }
    ntests++;
    if (HLTQ_LAST(hltq, test_data, entries) != &d3) {
	sudo_warnx_nodebug("FAIL: HLTQ_LAST(3 entries) doesn't return third element: got %p, expected %p", HLTQ_LAST(hltq, test_data, entries), &d3);
	errors++;
    }
    ntests++;

    if (HLTQ_NEXT(&d1, entries) != &d2) {
	sudo_warnx_nodebug("FAIL: HLTQ_NEXT(&d1) doesn't return &d2: got %p, expected %p", HLTQ_NEXT(&d1, entries), &d2);
	errors++;
    }
    ntests++;
    if (HLTQ_NEXT(&d2, entries) != &d3) {
	sudo_warnx_nodebug("FAIL: HLTQ_NEXT(&d2) doesn't return &d3: got %p, expected %p", HLTQ_NEXT(&d2, entries), &d3);
	errors++;
    }
    ntests++;
    if (HLTQ_NEXT(&d3, entries) != NULL) {
	sudo_warnx_nodebug("FAIL: HLTQ_NEXT(&d3) doesn't return NULL: got %p", HLTQ_NEXT(&d3, entries));
	errors++;
    }
    ntests++;

    if (HLTQ_PREV(&d1, test_data, entries) != NULL) {
	sudo_warnx_nodebug("FAIL: HLTQ_PREV(&d1) doesn't return NULL: got %p", HLTQ_PREV(&d1, test_data, entries));
	errors++;
    }
    ntests++;
    if (HLTQ_PREV(&d2, test_data, entries) != &d1) {
	sudo_warnx_nodebug("FAIL: HLTQ_PREV(&d2) doesn't return &d1: got %p, expected %p", HLTQ_PREV(&d2, test_data, entries), &d1);
	errors++;
    }
    ntests++;
    if (HLTQ_PREV(&d3, test_data, entries) != &d2) {
	sudo_warnx_nodebug("FAIL: HLTQ_PREV(&d3) doesn't return &d2: got %p, expected %p", HLTQ_PREV(&d3, test_data, entries), &d2);
	errors++;
    }
    ntests++;

    /* Test conversion to TAILQ. */
    HLTQ_TO_TAILQ(&tq, hltq, entries);

    if (TAILQ_FIRST(&tq) != &d1) {
	sudo_warnx_nodebug("FAIL: TAILQ_FIRST(&tq) doesn't return first element: got %p, expected %p", TAILQ_FIRST(&tq), &d1);
	errors++;
    }
    ntests++;
    if (TAILQ_LAST(&tq, test_data_list) != &d3) {
	sudo_warnx_nodebug("FAIL: TAILQ_LAST(&tq) doesn't return third element: got %p, expected %p", TAILQ_LAST(&tq, test_data_list), &d3);
	errors++;
    }
    ntests++;

    if (TAILQ_NEXT(&d1, entries) != &d2) {
	sudo_warnx_nodebug("FAIL: TAILQ_NEXT(&d1) doesn't return &d2: got %p, expected %p", TAILQ_NEXT(&d1, entries), &d2);
	errors++;
    }
    ntests++;
    if (TAILQ_NEXT(&d2, entries) != &d3) {
	sudo_warnx_nodebug("FAIL: TAILQ_NEXT(&d2) doesn't return &d3: got %p, expected %p", TAILQ_NEXT(&d2, entries), &d3);
	errors++;
    }
    ntests++;
    if (TAILQ_NEXT(&d3, entries) != NULL) {
	sudo_warnx_nodebug("FAIL: TAILQ_NEXT(&d3) doesn't return NULL: got %p", TAILQ_NEXT(&d3, entries));
	errors++;
    }
    ntests++;

    if (TAILQ_PREV(&d1, test_data_list, entries) != NULL) {
	sudo_warnx_nodebug("FAIL: TAILQ_PREV(&d1) doesn't return NULL: got %p", TAILQ_PREV(&d1, test_data_list, entries));
	errors++;
    }
    ntests++;
    if (TAILQ_PREV(&d2, test_data_list, entries) != &d1) {
	sudo_warnx_nodebug("FAIL: TAILQ_PREV(&d2) doesn't return &d1: got %p, expected %p", TAILQ_PREV(&d2, test_data_list, entries), &d1);
	errors++;
    }
    ntests++;
    if (TAILQ_PREV(&d3, test_data_list, entries) != &d2) {
	sudo_warnx_nodebug("FAIL: TAILQ_PREV(&d3) doesn't return &d2: got %p, expected %p", TAILQ_PREV(&d3, test_data_list, entries), &d2);
	errors++;
    }
    ntests++;

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }

    exit(errors);
}
