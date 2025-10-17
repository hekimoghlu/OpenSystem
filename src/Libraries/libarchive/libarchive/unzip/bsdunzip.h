/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#ifndef BSDUNZIP_H_INCLUDED
#define BSDUNZIP_H_INCLUDED

#if defined(PLATFORM_CONFIG_H)
/* Use hand-built config.h in environments that need it. */
#include PLATFORM_CONFIG_H
#else
/* Not having a config.h of some sort is a serious problem. */
#include "config.h"
#endif

#include <archive.h>
#include <archive_entry.h>

struct bsdunzip {
	/* Option parser state */
	int		  getopt_state;
	char		 *getopt_word;

	/* Miscellaneous state information */
	int		  argc;
	char		**argv;
	const char	 *argument;
};

enum {
	OPTION_NONE,
	OPTION_VERSION
};

int bsdunzip_getopt(struct bsdunzip *);

extern int bsdunzip_optind;

#endif
