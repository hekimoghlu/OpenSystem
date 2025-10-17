/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
 *	autod_lookup.c
 *
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * Portions Copyright 2007-2011 Apple Inc.
 */

#pragma ident	"@(#)autod_lookup.c	1.13	05/06/08 SMI"

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <syslog.h>
#include <errno.h>
#include <locale.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include "automount.h"

int
do_lookup1(const autofs_pathname mapname, const char *key,
    const autofs_pathname subdir, const autofs_opts mapopts,
    boolean_t isdirect, uid_t sendereuid, int *node_type)
{
	struct mapent *mapents = NULL;
	int err;
	bool_t isrestricted = hasrestrictopt(mapopts);

	/*
	 * call parser w default mount_access = TRUE
	 */
	mapents = parse_entry(key, mapname, mapopts, subdir, isdirect,
	    node_type, isrestricted, TRUE, &err);
	if (err) {
		/*
		 * The entry wasn't found in the map; err was set to
		 * the appropriate value by parse_entry().
		 *
		 * Now we indulge in a bit of hanky-panky.  If the
		 * name begins with an "=" then we assume that
		 * the name is an undocumented control message
		 * for the daemon.  This is accessible only
		 * to superusers.
		 */
		if (*key == '=' && sendereuid == 0) {
			if (isdigit(*(key + 1))) {
				/*
				 * If next character is a digit
				 * then set the trace level.
				 */
				trace = atoi(key + 1);
				trace_prt(1, "Automountd: trace level = %d\n",
				    trace);
			} else if (*(key + 1) == 'v') {
				/*
				 * If it's a "v" then
				 * toggle verbose mode.
				 */
				verbose = !verbose;
				trace_prt(1, "Automountd: verbose %s\n",
				    verbose ? "on" : "off");
			}
		}
		return err;
	}

	/*
	 * The entry was found in the map; err was set to 0
	 * by parse_entry().
	 */
	if (mapents) {
		free_mapent(mapents);
	}

	if (trace > 1) {
		trace_prt(1, "  do_lookup1: node_type=0x%08x error=%d\n",
		    *node_type, err);
	}
	return err;
}
