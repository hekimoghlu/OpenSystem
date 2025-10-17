/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#include <unistd.h>
#include <string.h>
#ifdef USE_PATHS_H
#include <paths.h>
#endif
#include <errno.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <argv.h>
#include <exec_command.h>

/* Application-specific. */

#define SPACE_TAB	" \t"

/* exec_command - exec command */

NORETURN exec_command(const char *command)
{
    ARGV   *argv;

    /*
     * Character filter. In this particular case, we allow space and tab in
     * addition to the regular character set.
     */
    static char ok_chars[] = "1234567890!@%-_=+:,./\
abcdefghijklmnopqrstuvwxyz\
ABCDEFGHIJKLMNOPQRSTUVWXYZ" SPACE_TAB;

    /*
     * See if this command contains any shell magic characters.
     */
    if (command[strspn(command, ok_chars)] == 0
	&& command[strspn(command, SPACE_TAB)] != 0) {

	/*
	 * No shell meta characters found, so we can try to avoid the overhead
	 * of running a shell. Just split the command on whitespace and exec
	 * the result directly.
	 */
	argv = argv_split(command, SPACE_TAB);
	(void) execvp(argv->argv[0], argv->argv);

	/*
	 * Auch. Perhaps they're using some shell built-in command.
	 */
	if (errno != ENOENT || strchr(argv->argv[0], '/') != 0)
	    msg_fatal("execvp %s: %m", argv->argv[0]);

	/*
	 * Not really necessary, but...
	 */
	argv_free(argv);
    }

    /*
     * Pass the command to a shell.
     */
    (void) execl(_PATH_BSHELL, "sh", "-c", command, (char *) 0);
    msg_fatal("execl %s: %m", _PATH_BSHELL);
}

#ifdef TEST

 /*
  * Yet another proof-of-concept test program.
  */
#include <vstream.h>
#include <msg_vstream.h>

int     main(int argc, char **argv)
{
    msg_vstream_init(argv[0], VSTREAM_ERR);
    if (argc != 2)
	msg_fatal("usage: %s 'command'", argv[0]);
    exec_command(argv[1]);
}

#endif
