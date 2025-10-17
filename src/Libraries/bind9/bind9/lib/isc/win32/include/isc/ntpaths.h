/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
/* $Id: ntpaths.h,v 1.20 2009/07/14 22:54:57 each Exp $ */

/*
 * Windows-specific path definitions
 * These routines are used to set up and return system-specific path
 * information about the files enumerated in NtPaths
 */

#ifndef ISC_NTPATHS_H
#define ISC_NTPATHS_H

#include <isc/lang.h>

/*
 * Index of paths needed
 */
enum NtPaths {
	NAMED_CONF_PATH,
	LWRES_CONF_PATH,
	RESOLV_CONF_PATH,
	RNDC_CONF_PATH,
	NAMED_PID_PATH,
	LWRESD_PID_PATH,
	LOCAL_STATE_DIR,
	SYS_CONF_DIR,
	RNDC_KEY_PATH,
	SESSION_KEY_PATH
};

/*
 * Define macros to get the path of the config files
 */
#define NAMED_CONFFILE isc_ntpaths_get(NAMED_CONF_PATH)
#define RNDC_CONFFILE isc_ntpaths_get(RNDC_CONF_PATH)
#define RNDC_KEYFILE isc_ntpaths_get(RNDC_KEY_PATH)
#define SESSION_KEYFILE isc_ntpaths_get(SESSION_KEY_PATH)
#define RESOLV_CONF isc_ntpaths_get(RESOLV_CONF_PATH)

/*
 * Information about where the files are on disk
 */
#define NS_LOCALSTATEDIR	"/dns/bin"
#define NS_SYSCONFDIR		"/dns/etc"

ISC_LANG_BEGINDECLS

void
isc_ntpaths_init(void);

char *
isc_ntpaths_get(int);

ISC_LANG_ENDDECLS

#endif /* ISC_NTPATHS_H */
