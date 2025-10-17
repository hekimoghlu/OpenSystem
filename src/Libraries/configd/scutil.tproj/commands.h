/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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
 * Modification History
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * November 9, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _COMMANDS_H
#define _COMMANDS_H

#include <sys/cdefs.h>

typedef const struct {
	char	*cmd;
	int	minArgs;
	int	maxArgs;
	void	(*func)(int argc, char * const argv[]);
	int	group;
	int	ctype;	/* -1==normal/hidden, 0==normal, 1==limited, 2==private */
	char	*usage;
} cmdInfo;

extern const cmdInfo	commands_store[];
extern const int	nCommands_store;

extern const cmdInfo	commands_net[];
extern const int	nCommands_net;

extern const cmdInfo	commands_prefs[];
extern const int	nCommands_prefs;

extern const cmdInfo	*commands;
extern int		nCommands;
extern Boolean		enablePrivateAPI;
extern Boolean		termRequested;

__BEGIN_DECLS

void	do_command		(int argc, char * const argv[]);
void	do_help			(int argc, char * const argv[]);
void	do_quit			(int argc, char * const argv[]);
void	do_readFile		(int argc, char * const argv[]);

__END_DECLS

#endif /* !_COMMANDS_H */
