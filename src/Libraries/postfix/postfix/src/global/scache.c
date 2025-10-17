/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <vstring_vstream.h>
#include <argv.h>
#include <events.h>

/* Global library. */

#include <scache.h>

#ifdef TEST

 /*
  * Driver program for cache regression tests. Although all variants are
  * relatively simple to verify by hand for single session storage, more
  * sophisticated instrumentation is needed to demonstrate that the
  * multi-session cache manager properly handles collisions in the time
  * domain and in all the name space domains.
  */
static SCACHE *scache;
static VSTRING *endp_prop;
static VSTRING *dest_prop;
static int verbose_level = 3;

 /*
  * Cache mode lookup table. We don't do the client-server variant because
  * that drags in a ton of configuration junk; the client-server protocol is
  * relatively easy to verify manually.
  */
struct cache_type {
    char   *mode;
    SCACHE *(*create) (void);
};

static struct cache_type cache_types[] = {
    "single", scache_single_create,
    "multi", scache_multi_create,
    0,
};

#define STR(x) vstring_str(x)

/* cache_type - select session cache type */

static void cache_type(ARGV *argv)
{
    struct cache_type *cp;

    if (argv->argc != 2) {
	msg_error("usage: %s mode", argv->argv[0]);
	return;
    }
    if (scache != 0)
	scache_free(scache);
    for (cp = cache_types; cp->mode != 0; cp++) {
	if (strcmp(cp->mode, argv->argv[1]) == 0) {
	    scache = cp->create();
	    return;
	}
    }
    msg_error("unknown cache type: %s", argv->argv[1]);
}

/* handle_events - handle events while time advances */

static void handle_events(ARGV *argv)
{
    int     delay;
    time_t  before;
    time_t  after;

    if (argv->argc != 2 || (delay = atoi(argv->argv[1])) <= 0) {
	msg_error("usage: %s time", argv->argv[0]);
	return;
    }
    before = event_time();
    event_drain(delay);
    after = event_time();
    if (after < before + delay)
	sleep(before + delay - after);
}

/* save_endp - save endpoint->session binding */

static void save_endp(ARGV *argv)
{
    int     ttl;
    int     fd;

    if (argv->argc != 5
	|| (ttl = atoi(argv->argv[1])) <= 0
	|| (fd = atoi(argv->argv[4])) <= 0) {
	msg_error("usage: save_endp ttl endpoint endp_props fd");
	return;
    }
    if (DUP2(0, fd) < 0)
	msg_fatal("dup2(0, %d): %m", fd);
    scache_save_endp(scache, ttl, argv->argv[2], argv->argv[3], fd);
}

/* find_endp - find endpoint->session binding */

static void find_endp(ARGV *argv)
{
    int     fd;

    if (argv->argc != 2) {
	msg_error("usage: find_endp endpoint");
	return;
    }
    if ((fd = scache_find_endp(scache, argv->argv[1], endp_prop)) >= 0)
	close(fd);
}

/* save_dest - save destination->endpoint binding */

static void save_dest(ARGV *argv)
{
    int     ttl;

    if (argv->argc != 5 || (ttl = atoi(argv->argv[1])) <= 0) {
	msg_error("usage: save_dest ttl destination dest_props endpoint");
	return;
    }
    scache_save_dest(scache, ttl, argv->argv[2], argv->argv[3], argv->argv[4]);
}

/* find_dest - find destination->endpoint->session binding */

static void find_dest(ARGV *argv)
{
    int     fd;

    if (argv->argc != 2) {
	msg_error("usage: find_dest destination");
	return;
    }
    if ((fd = scache_find_dest(scache, argv->argv[1], dest_prop, endp_prop)) >= 0)
	close(fd);
}

/* verbose - adjust noise level during cache manipulation */

static void verbose(ARGV *argv)
{
    int     level;

    if (argv->argc != 2 || (level = atoi(argv->argv[1])) < 0) {
	msg_error("usage: verbose level");
	return;
    }
    verbose_level = level;
}

 /*
  * The command lookup table.
  */
struct action {
    char   *command;
    void    (*action) (ARGV *);
    int     flags;
};

#define FLAG_NEED_CACHE	(1<<0)

static void help(ARGV *);

static struct action actions[] = {
    "cache_type", cache_type, 0,
    "save_endp", save_endp, FLAG_NEED_CACHE,
    "find_endp", find_endp, FLAG_NEED_CACHE,
    "save_dest", save_dest, FLAG_NEED_CACHE,
    "find_dest", find_dest, FLAG_NEED_CACHE,
    "sleep", handle_events, 0,
    "verbose", verbose, 0,
    "?", help, 0,
    0,
};

/* help - list commands */

static void help(ARGV *argv)
{
    struct action *ap;

    vstream_printf("commands:");
    for (ap = actions; ap->command != 0; ap++)
	vstream_printf(" %s", ap->command);
    vstream_printf("\n");
    vstream_fflush(VSTREAM_OUT);
}

/* get_buffer - prompt for input or log input */

static int get_buffer(VSTRING *buf, VSTREAM *fp, int interactive)
{
    int     status;

    if (interactive) {
	vstream_printf("> ");
	vstream_fflush(VSTREAM_OUT);
    }
    if ((status = vstring_get_nonl(buf, fp)) != VSTREAM_EOF) {
	if (!interactive) {
	    vstream_printf(">>> %s\n", STR(buf));
	    vstream_fflush(VSTREAM_OUT);
	}
    }
    return (status);
}

/* at last, the main program */

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *buf = vstring_alloc(1);
    ARGV   *argv;
    struct action *ap;
    int     interactive = isatty(0);

    endp_prop = vstring_alloc(1);
    dest_prop = vstring_alloc(1);

    vstream_fileno(VSTREAM_ERR) = 1;

    while (get_buffer(buf, VSTREAM_IN, interactive) != VSTREAM_EOF) {
	argv = argv_split(STR(buf), CHARS_SPACE);
	if (argv->argc > 0 && argv->argv[0][0] != '#') {
	    msg_verbose = verbose_level;
	    for (ap = actions; ap->command != 0; ap++) {
		if (strcmp(ap->command, argv->argv[0]) == 0) {
		    if ((ap->flags & FLAG_NEED_CACHE) != 0 && scache == 0)
			msg_error("no session cache");
		    else
			ap->action(argv);
		    break;
		}
	    }
	    msg_verbose = 0;
	    if (ap->command == 0)
		msg_error("bad command: %s", argv->argv[0]);
	}
	argv_free(argv);
    }
    scache_free(scache);
    vstring_free(endp_prop);
    vstring_free(dest_prop);
    vstring_free(buf);
    exit(0);
}

#endif
