/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#ifndef _SECURITYTOOL_H_
#define _SECURITYTOOL_H_

#include <sys/cdefs.h>

__BEGIN_DECLS

typedef int(*command_func)(int argc, char * const *argv);

/* Entry in commands array for a command. */
typedef struct command
{
    const char *c_name;    /* name of the command. */
    command_func c_func;   /* function to execute the command. */
    const char *c_usage;   /* usage sting for command. */
    const char *c_help;    /* help string for (or description of) command. */
} command;

/*
 * The command array itself.
 * Add commands here at will.
 * Matching is done on a prefix basis.  The first command in the array
 * gets matched first.
 */
extern const command commands[];

/* Our one builtin command.
 */
int help(int argc, char * const *argv);
    
/* If 1 attempt to be as quiet as possible. */
extern int do_quiet;

/* If 1 attempt to be as verbose as possible. */
extern int do_verbose;

__END_DECLS

#endif /*  _SECURITY_H_ */
