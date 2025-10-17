/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
/* $Id: syslog.c,v 1.8 2007/09/13 04:45:18 each Exp $ */

/*! \file */

#include <config.h>

#include <stdlib.h>
#include <syslog.h>

#include <isc/result.h>
#include <isc/string.h>
#include <isc/syslog.h>
#include <isc/util.h>

static struct dsn_c_pvt_sfnt {
	int val;
	const char *strval;
} facilities[] = {
	{ LOG_KERN,			"kern" },
	{ LOG_USER,			"user" },
	{ LOG_MAIL,			"mail" },
	{ LOG_DAEMON,			"daemon" },
	{ LOG_AUTH,			"auth" },
	{ LOG_SYSLOG,			"syslog" },
	{ LOG_LPR,			"lpr" },
#ifdef LOG_NEWS
	{ LOG_NEWS,			"news" },
#endif
#ifdef LOG_UUCP
	{ LOG_UUCP,			"uucp" },
#endif
#ifdef LOG_CRON
	{ LOG_CRON,			"cron" },
#endif
#ifdef LOG_AUTHPRIV
	{ LOG_AUTHPRIV,			"authpriv" },
#endif
#ifdef LOG_FTP
	{ LOG_FTP,			"ftp" },
#endif
	{ LOG_LOCAL0,			"local0"},
	{ LOG_LOCAL1,			"local1"},
	{ LOG_LOCAL2,			"local2"},
	{ LOG_LOCAL3,			"local3"},
	{ LOG_LOCAL4,			"local4"},
	{ LOG_LOCAL5,			"local5"},
	{ LOG_LOCAL6,			"local6"},
	{ LOG_LOCAL7,			"local7"},
	{ 0,				NULL }
};

isc_result_t
isc_syslog_facilityfromstring(const char *str, int *facilityp) {
	int i;

	REQUIRE(str != NULL);
	REQUIRE(facilityp != NULL);

	for (i = 0; facilities[i].strval != NULL; i++) {
		if (strcasecmp(facilities[i].strval, str) == 0) {
			*facilityp = facilities[i].val;
			return (ISC_R_SUCCESS);
		}
	}
	return (ISC_R_NOTFOUND);

}
