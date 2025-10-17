/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
/* $OpenLDAP$ */
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 1998-2011 The OpenLDAP Foundation.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in the file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */
/* Portions Copyright (c) 1995 Regents of the University of Michigan.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that this notice is preserved and that due credit is given
 * to the University of Michigan at Ann Arbor. The name of the University
 * may not be used to endorse or promote products derived from this
 * software without specific prior written permission. This software
 * is provided ``as is'' without express or implied warranty.
 */
/* ACKNOWLEDGEMENTS:
 * This work was originally developed by the University of Michigan
 * (as part of U-MICH LDAP).
 */

#include "portable.h"

#include <stdio.h>

#include <ac/socket.h>
#include <ac/string.h>
#include <ac/time.h>

#include "slap.h"
#include "back-passwd.h"
#include "config.h"

static ConfigTable passwdcfg[] = {
	{ "file", "filename", 2, 2, 0,
#ifdef HAVE_SETPWFILE
		ARG_STRING|ARG_OFFSET, NULL,
#else
		ARG_IGNORED, NULL,
#endif
		"( OLcfgDbAt:9.1 NAME 'olcPasswdFile' "
			"DESC 'File containing passwd records' "
			"EQUALITY caseExactMatch "
			"SYNTAX OMsDirectoryString SINGLE-VALUE )", NULL, NULL },
	{ NULL, NULL, 0, 0, 0, ARG_IGNORED,
		NULL, NULL, NULL, NULL }
};

static ConfigOCs passwdocs[] = {
	{ "( OLcfgDbOc:9.1 "
			"NAME 'olcPasswdConfig' "
			"DESC 'Passwd backend configuration' "
			"SUP olcDatabaseConfig "
			"MAY olcPasswdFile )",
			Cft_Database, passwdcfg },
	{ NULL, 0, NULL }
};

int
passwd_back_init_cf( BackendInfo *bi )
{
	bi->bi_cf_ocs = passwdocs;
	return config_register_schema( passwdcfg, passwdocs );
}
