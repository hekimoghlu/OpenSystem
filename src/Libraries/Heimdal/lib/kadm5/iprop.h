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
/* $Id$ */

#ifndef __IPROP_H__
#define __IPROP_H__

#include "kadm5_locl.h"
#include <getarg.h>
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif
#ifdef HAVE_UTIL_H
#include <util.h>
#endif

#include <parse_time.h>

#define IPROP_VERSION "iprop-0.0"

#define IPROP_NAME "iprop"

#define IPROP_SERVICE "iprop"

#define IPROP_PORT 2121

enum iprop_cmd { I_HAVE = 1,
		 FOR_YOU = 2,
		 TELL_YOU_EVERYTHING = 3,
		 ONE_PRINC = 4,
		 NOW_YOU_HAVE = 5,
		 ARE_YOU_THERE = 6,
		 I_AM_HERE = 7
};

extern sig_atomic_t exit_flag;
void setup_signal(void);

#endif /* __IPROP_H__ */
