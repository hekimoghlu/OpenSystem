/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#ifndef __CONFIG_PARAMS_H__
#define __CONFIG_PARAMS_H__

//
// Parameters set from the configuration file.
//

#define MAX_LINE 2048		/* Maximum chars allowed for the host list (in passive mode) */
#define MAX_HOST_LIST 64000
#define MAX_ACTIVE_LIST 10

struct active_pars
{
	char address[MAX_LINE + 1];	// keeps the network address (either numeric or literal) to of the active client
	char port[MAX_LINE + 1];	// keeps the network port to bind to
	int ai_family;			// address faimly to use
};

extern char hostlist[MAX_HOST_LIST + 1];	//!< Keeps the list of the hosts that are allowed to connect to this server
extern struct active_pars activelist[MAX_ACTIVE_LIST];		//!< Keeps the list of the hosts (host, port) on which I want to connect to (active mode)
extern int nullAuthAllowed;			//!< '1' if we permit NULL authentication, '0' otherwise
extern char loadfile[MAX_LINE + 1];		//!< Name of the file from which we have to load the configuration

#endif
