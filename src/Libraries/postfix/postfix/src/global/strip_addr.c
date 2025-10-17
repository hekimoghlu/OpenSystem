/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#include <string.h>

/* Utility library. */

#include <mymalloc.h>

/* Global library. */

#include <split_addr.h>
#include <strip_addr.h>

/* strip_addr - strip extension from address */

char   *strip_addr_internal(const char *full, char **extension,
			            const char *delimiter_set)
{
    char   *ratsign;
    char   *extent;
    char   *saved_ext;
    char   *stripped;

    /*
     * A quick test to eliminate inputs without delimiter anywhere.
     */
    if (*delimiter_set == 0 || full[strcspn(full, delimiter_set)] == 0) {
	stripped = saved_ext = 0;
    } else {
	stripped = mystrdup(full);
	if ((ratsign = strrchr(stripped, '@')) != 0)
	    *ratsign = 0;
	if ((extent = split_addr(stripped, delimiter_set)) != 0) {
	    extent -= 1;
	    if (extension) {
		*extent = full[strlen(stripped)];
		saved_ext = mystrdup(extent);
		*extent = 0;
	    } else
		saved_ext = 0;
	    if (ratsign != 0) {
		*ratsign = '@';
		memmove(extent, ratsign, strlen(ratsign) + 1);
	    }
	} else {
	    myfree(stripped);
	    stripped = saved_ext = 0;
	}
    }
    if (extension)
	*extension = saved_ext;
    return (stripped);
}

#ifdef TEST

#include <msg.h>
#include <mail_params.h>

char   *var_double_bounce_sender = DEF_DOUBLE_BOUNCE;

int     main(int unused_argc, char **unused_argv)
{
    char   *extension;
    char   *stripped;
    char   *delim = "+-";

#define NO_DELIM	""

    /*
     * Incredible. This function takes only three arguments, and the tests
     * already take more lines of code than the code being tested.
     */
    stripped = strip_addr_internal("foo", (char **) 0, NO_DELIM);
    if (stripped != 0)
	msg_panic("strip_addr botch 1");

    stripped = strip_addr_internal("foo", &extension, NO_DELIM);
    if (stripped != 0)
	msg_panic("strip_addr botch 2");
    if (extension != 0)
	msg_panic("strip_addr botch 3");

    stripped = strip_addr_internal("foo", (char **) 0, delim);
    if (stripped != 0)
	msg_panic("strip_addr botch 4");

    stripped = strip_addr_internal("foo", &extension, delim);
    if (stripped != 0)
	msg_panic("strip_addr botch 5");
    if (extension != 0)
	msg_panic("strip_addr botch 6");

    stripped = strip_addr_internal("foo@bar", (char **) 0, NO_DELIM);
    if (stripped != 0)
	msg_panic("strip_addr botch 7");

    stripped = strip_addr_internal("foo@bar", &extension, NO_DELIM);
    if (stripped != 0)
	msg_panic("strip_addr botch 8");
    if (extension != 0)
	msg_panic("strip_addr botch 9");

    stripped = strip_addr_internal("foo@bar", (char **) 0, delim);
    if (stripped != 0)
	msg_panic("strip_addr botch 10");

    stripped = strip_addr_internal("foo@bar", &extension, delim);
    if (stripped != 0)
	msg_panic("strip_addr botch 11");
    if (extension != 0)
	msg_panic("strip_addr botch 12");

    stripped = strip_addr_internal("foo-ext", (char **) 0, NO_DELIM);
    if (stripped != 0)
	msg_panic("strip_addr botch 13");

    stripped = strip_addr_internal("foo-ext", &extension, NO_DELIM);
    if (stripped != 0)
	msg_panic("strip_addr botch 14");
    if (extension != 0)
	msg_panic("strip_addr botch 15");

    stripped = strip_addr_internal("foo-ext", (char **) 0, delim);
    if (stripped == 0)
	msg_panic("strip_addr botch 16");
    msg_info("wanted:    foo-ext -> %s", "foo");
    msg_info("strip_addr foo-ext -> %s", stripped);
    myfree(stripped);

    stripped = strip_addr_internal("foo-ext", &extension, delim);
    if (stripped == 0)
	msg_panic("strip_addr botch 17");
    if (extension == 0)
	msg_panic("strip_addr botch 18");
    msg_info("wanted:    foo-ext -> %s %s", "foo", "-ext");
    msg_info("strip_addr foo-ext -> %s %s", stripped, extension);
    myfree(stripped);
    myfree(extension);

    stripped = strip_addr_internal("foo-ext@bar", (char **) 0, NO_DELIM);
    if (stripped != 0)
	msg_panic("strip_addr botch 19");

    stripped = strip_addr_internal("foo-ext@bar", &extension, NO_DELIM);
    if (stripped != 0)
	msg_panic("strip_addr botch 20");
    if (extension != 0)
	msg_panic("strip_addr botch 21");

    stripped = strip_addr_internal("foo-ext@bar", (char **) 0, delim);
    if (stripped == 0)
	msg_panic("strip_addr botch 22");
    msg_info("wanted:    foo-ext@bar -> %s", "foo@bar");
    msg_info("strip_addr foo-ext@bar -> %s", stripped);
    myfree(stripped);

    stripped = strip_addr_internal("foo-ext@bar", &extension, delim);
    if (stripped == 0)
	msg_panic("strip_addr botch 23");
    if (extension == 0)
	msg_panic("strip_addr botch 24");
    msg_info("wanted:    foo-ext@bar -> %s %s", "foo@bar", "-ext");
    msg_info("strip_addr foo-ext@bar -> %s %s", stripped, extension);
    myfree(stripped);
    myfree(extension);

    stripped = strip_addr_internal("foo+ext@bar", &extension, delim);
    if (stripped == 0)
	msg_panic("strip_addr botch 25");
    if (extension == 0)
	msg_panic("strip_addr botch 26");
    msg_info("wanted:    foo+ext@bar -> %s %s", "foo@bar", "+ext");
    msg_info("strip_addr foo+ext@bar -> %s %s", stripped, extension);
    myfree(stripped);
    myfree(extension);

    stripped = strip_addr_internal("foo bar+ext", &extension, delim);
    if (stripped == 0)
	msg_panic("strip_addr botch 27");
    if (extension == 0)
	msg_panic("strip_addr botch 28");
    msg_info("wanted:    foo bar+ext -> %s %s", "foo bar", "+ext");
    msg_info("strip_addr foo bar+ext -> %s %s", stripped, extension);
    myfree(stripped);
    myfree(extension);

    return (0);
}

#endif
