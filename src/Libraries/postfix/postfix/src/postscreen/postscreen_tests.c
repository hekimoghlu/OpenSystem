/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 10, 2025.
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
/* System library. */

#include <sys_defs.h>
#include <stdio.h>			/* sscanf */
#include <stdlib.h>			/* strtoul */

/* Utility library. */

#include <msg.h>
#include <name_code.h>

/* Global library. */

#include <mail_params.h>

/* Application-specific. */

#include <postscreen.h>

 /*
  * Kludge to detect if some test is enabled.
  */
#define PSC_PREGR_TEST_ENABLE()	(*var_psc_pregr_banner != 0)
#define PSC_DNSBL_TEST_ENABLE()	(*var_psc_dnsbl_sites != 0)

 /*
  * Format of a persistent cache entry (which is almost but not quite the
  * same as the in-memory representation).
  * 
  * Each cache entry has one time stamp for each test.
  * 
  * - A time stamp of PSC_TIME_STAMP_INVALID must never appear in the cache. It
  * is reserved for in-memory objects that are still being initialized.
  * 
  * - A time stamp of PSC_TIME_STAMP_NEW indicates that the test never passed.
  * Postscreen will log the client with "pass new" when it passes the final
  * test.
  * 
  * - A time stamp of PSC_TIME_STAMP_DISABLED indicates that the test never
  * passed, and that the test was disabled when the cache entry was written.
  * 
  * - Otherwise, the test was passed, and the time stamp indicates when that
  * test result expires.
  * 
  * A cache entry is expired when the time stamps of all passed tests are
  * expired.
  */

/* psc_new_tests - initialize new test results from scratch */

void    psc_new_tests(PSC_STATE *state)
{
    time_t *expire_time = state->client_info->expire_time;

    /*
     * Give all tests a PSC_TIME_STAMP_NEW time stamp, so that we can later
     * recognize cache entries that haven't passed all enabled tests. When we
     * write a cache entry to the database, any new-but-disabled tests will
     * get a PSC_TIME_STAMP_DISABLED time stamp.
     */
    expire_time[PSC_TINDX_PREGR] = PSC_TIME_STAMP_NEW;
    expire_time[PSC_TINDX_DNSBL] = PSC_TIME_STAMP_NEW;
    expire_time[PSC_TINDX_PIPEL] = PSC_TIME_STAMP_NEW;
    expire_time[PSC_TINDX_NSMTP] = PSC_TIME_STAMP_NEW;
    expire_time[PSC_TINDX_BARLF] = PSC_TIME_STAMP_NEW;

    /*
     * Determine what tests need to be completed.
     */
    psc_todo_tests(state, PSC_TIME_STAMP_NEW + 1);
}

/* psc_parse_tests - parse test results from cache */

void    psc_parse_tests(PSC_STATE *state,
			        const char *stamp_str,
			        time_t time_value)
{
    const char *start = stamp_str;
    char   *cp;
    time_t *time_stamps = state->client_info->expire_time;
    time_t *sp;

    /*
     * Parse the cache entry, and allow for older postscreen versions that
     * implemented fewer tests. We pretend that the newer tests were disabled
     * at the time that the cache entry was written.
     */
    for (sp = time_stamps; sp < time_stamps + PSC_TINDX_COUNT; sp++) {
	*sp = strtoul(start, &cp, 10);
	if (*start == 0 || (*cp != '\0' && *cp != ';') || errno == ERANGE)
	    *sp = PSC_TIME_STAMP_DISABLED;
	if (msg_verbose)
	    msg_info("%s -> %lu", start, (unsigned long) *sp);
	if (*cp == ';')
	    start = cp + 1;
	else
	    start = cp;
    }

    /*
     * Determine what tests need to be completed.
     */
    psc_todo_tests(state, time_value);
}

/* psc_todo_tests - determine what tests to perform */

void    psc_todo_tests(PSC_STATE *state, time_t time_value)
{
    time_t *expire_time = state->client_info->expire_time;
    time_t *sp;

    /*
     * Reset all per-session flags.
     */
    state->flags = 0;

    /*
     * Flag the tests as "new" when the cache entry has fields for all
     * enabled tests, but the remote SMTP client has not yet passed all those
     * tests.
     */
    for (sp = expire_time; sp < expire_time + PSC_TINDX_COUNT; sp++) {
	if (*sp == PSC_TIME_STAMP_NEW)
	    state->flags |= PSC_STATE_FLAG_NEW;
    }

    /*
     * Don't flag disabled tests as "todo", because there would be no way to
     * make those bits go away.
     */
    if (PSC_PREGR_TEST_ENABLE() && time_value > expire_time[PSC_TINDX_PREGR])
	state->flags |= PSC_STATE_FLAG_PREGR_TODO;
    if (PSC_DNSBL_TEST_ENABLE() && time_value > expire_time[PSC_TINDX_DNSBL])
	state->flags |= PSC_STATE_FLAG_DNSBL_TODO;
    if (var_psc_pipel_enable && time_value > expire_time[PSC_TINDX_PIPEL])
	state->flags |= PSC_STATE_FLAG_PIPEL_TODO;
    if (var_psc_nsmtp_enable && time_value > expire_time[PSC_TINDX_NSMTP])
	state->flags |= PSC_STATE_FLAG_NSMTP_TODO;
    if (var_psc_barlf_enable && time_value > expire_time[PSC_TINDX_BARLF])
	state->flags |= PSC_STATE_FLAG_BARLF_TODO;

    /*
     * If any test has expired, proactively refresh tests that will expire
     * soon. This can increase the occurrence of client-visible delays, but
     * avoids questions about why a client can pass some test and then fail
     * within seconds. The proactive refresh time is really a surrogate for
     * the user's curiosity level, and therefore hard to choose optimally.
     */
#ifdef VAR_PSC_REFRESH_TIME
    if ((state->flags & PSC_STATE_MASK_ANY_TODO) != 0
	&& var_psc_refresh_time > 0) {
	time_t  refresh_time = time_value + var_psc_refresh_time;

	if (PSC_PREGR_TEST_ENABLE() && refresh_time > expire_time[PSC_TINDX_PREGR])
	    state->flags |= PSC_STATE_FLAG_PREGR_TODO;
	if (PSC_DNSBL_TEST_ENABLE() && refresh_time > expire_time[PSC_TINDX_DNSBL])
	    state->flags |= PSC_STATE_FLAG_DNSBL_TODO;
	if (var_psc_pipel_enable && refresh_time > expire_time[PSC_TINDX_PIPEL])
	    state->flags |= PSC_STATE_FLAG_PIPEL_TODO;
	if (var_psc_nsmtp_enable && refresh_time > expire_time[PSC_TINDX_NSMTP])
	    state->flags |= PSC_STATE_FLAG_NSMTP_TODO;
	if (var_psc_barlf_enable && refresh_time > expire_time[PSC_TINDX_BARLF])
	    state->flags |= PSC_STATE_FLAG_BARLF_TODO;
    }
#endif

    /*
     * Gratuitously make postscreen logging more useful by turning on all
     * enabled pre-handshake tests when any pre-handshake test is turned on.
     * 
     * XXX Don't enable PREGREET gratuitously before the test expires. With a
     * short TTL for DNSBL whitelisting, turning on PREGREET would force a
     * full postscreen_greet_wait too frequently.
     */
#if 0
    if (state->flags & PSC_STATE_MASK_EARLY_TODO) {
	if (PSC_PREGR_TEST_ENABLE())
	    state->flags |= PSC_STATE_FLAG_PREGR_TODO;
	if (PSC_DNSBL_TEST_ENABLE())
	    state->flags |= PSC_STATE_FLAG_DNSBL_TODO;
    }
#endif
}

/* psc_print_tests - print postscreen cache record */

char   *psc_print_tests(VSTRING *buf, PSC_STATE *state)
{
    const char *myname = "psc_print_tests";
    time_t *expire_time = state->client_info->expire_time;

    /*
     * Sanity check.
     */
    if ((state->flags & PSC_STATE_MASK_ANY_UPDATE) == 0)
	msg_panic("%s: attempt to save a no-update record", myname);

    /*
     * Give disabled tests a dummy time stamp so that we don't log a client
     * with "pass new" when some disabled test becomes enabled at some later
     * time.
     */
    if (PSC_PREGR_TEST_ENABLE() == 0 && expire_time[PSC_TINDX_PREGR] == PSC_TIME_STAMP_NEW)
	expire_time[PSC_TINDX_PREGR] = PSC_TIME_STAMP_DISABLED;
    if (PSC_DNSBL_TEST_ENABLE() == 0 && expire_time[PSC_TINDX_DNSBL] == PSC_TIME_STAMP_NEW)
	expire_time[PSC_TINDX_DNSBL] = PSC_TIME_STAMP_DISABLED;
    if (var_psc_pipel_enable == 0 && expire_time[PSC_TINDX_PIPEL] == PSC_TIME_STAMP_NEW)
	expire_time[PSC_TINDX_PIPEL] = PSC_TIME_STAMP_DISABLED;
    if (var_psc_nsmtp_enable == 0 && expire_time[PSC_TINDX_NSMTP] == PSC_TIME_STAMP_NEW)
	expire_time[PSC_TINDX_NSMTP] = PSC_TIME_STAMP_DISABLED;
    if (var_psc_barlf_enable == 0 && expire_time[PSC_TINDX_BARLF] == PSC_TIME_STAMP_NEW)
	expire_time[PSC_TINDX_BARLF] = PSC_TIME_STAMP_DISABLED;

    vstring_sprintf(buf, "%lu;%lu;%lu;%lu;%lu",
		    (unsigned long) expire_time[PSC_TINDX_PREGR],
		    (unsigned long) expire_time[PSC_TINDX_DNSBL],
		    (unsigned long) expire_time[PSC_TINDX_PIPEL],
		    (unsigned long) expire_time[PSC_TINDX_NSMTP],
		    (unsigned long) expire_time[PSC_TINDX_BARLF]);
    return (STR(buf));
}

/* psc_print_grey_key - print postscreen cache record */

char   *psc_print_grey_key(VSTRING *buf, const char *client,
			           const char *helo, const char *sender,
			           const char *rcpt)
{
    return (STR(vstring_sprintf(buf, "%s/%s/%s/%s",
				client, helo, sender, rcpt)));
}

/* psc_test_name - map test index to symbolic name */

const char *psc_test_name(int tindx)
{
    const char *myname = "psc_test_name";
    const NAME_CODE test_name_map[] = {
	PSC_TNAME_PREGR, PSC_TINDX_PREGR,
	PSC_TNAME_DNSBL, PSC_TINDX_DNSBL,
	PSC_TNAME_PIPEL, PSC_TINDX_PIPEL,
	PSC_TNAME_NSMTP, PSC_TINDX_NSMTP,
	PSC_TNAME_BARLF, PSC_TINDX_BARLF,
	0, -1,
    };
    const char *result;

    if ((result = str_name_code(test_name_map, tindx)) == 0)
	msg_panic("%s: bad index %d", myname, tindx);
    return (result);
}
