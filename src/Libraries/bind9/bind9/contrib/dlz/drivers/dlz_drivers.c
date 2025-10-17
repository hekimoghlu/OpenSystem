/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
/* $Id: dlz_drivers.c,v 1.4 2011/03/10 04:36:16 each Exp $ */

/*! \file */

#include <config.h>

#include <isc/result.h>

/*
 * Pull in declarations for this module's functions.
 */

#include <dlz/dlz_drivers.h>

/*
 * Pull in driver-specific stuff.
 */

#ifdef DLZ_STUB
#include <dlz/dlz_stub_driver.h>
#endif

#ifdef DLZ_POSTGRES
#include <dlz/dlz_postgres_driver.h>
#endif

#ifdef DLZ_MYSQL
#include <dlz/dlz_mysql_driver.h>
#endif

#ifdef DLZ_FILESYSTEM
#include <dlz/dlz_filesystem_driver.h>
#endif

#ifdef DLZ_BDB
#include <dlz/dlz_bdb_driver.h>
#include <dlz/dlz_bdbhpt_driver.h>
#endif

#ifdef DLZ_LDAP
#include <dlz/dlz_ldap_driver.h>
#endif

#ifdef DLZ_ODBC
#include <dlz/dlz_odbc_driver.h>
#endif

/*%
 * Call init functions for all relevant DLZ drivers.
 */

isc_result_t
dlz_drivers_init(void) {

	isc_result_t result = ISC_R_SUCCESS;

#ifdef DLZ_STUB
	result = dlz_stub_init();
	if (result != ISC_R_SUCCESS)
		return (result);
#endif

#ifdef DLZ_POSTGRES
	result = dlz_postgres_init();
	if (result != ISC_R_SUCCESS)
		return (result);
#endif

#ifdef DLZ_MYSQL
	result = dlz_mysql_init();
	if (result != ISC_R_SUCCESS)
		return (result);
#endif

#ifdef DLZ_FILESYSTEM
	result = dlz_fs_init();
	if (result != ISC_R_SUCCESS)
		return (result);
#endif

#ifdef DLZ_BDB
	result = dlz_bdb_init();
	if (result != ISC_R_SUCCESS)
		return (result);
	result = dlz_bdbhpt_init();
	if (result != ISC_R_SUCCESS)
		return (result);
#endif

#ifdef DLZ_LDAP
	result = dlz_ldap_init();
	if (result != ISC_R_SUCCESS)
		return (result);
#endif

#ifdef DLZ_ODBC
	result = dlz_odbc_init();
	if (result != ISC_R_SUCCESS)
		return (result);
#endif

	return (result);
}

/*%
 * Call shutdown functions for all relevant DLZ drivers.
 */

void
dlz_drivers_clear(void) {

#ifdef DLZ_STUB
	dlz_stub_clear();
#endif

#ifdef DLZ_POSTGRES
        dlz_postgres_clear();
#endif

#ifdef DLZ_MYSQL
 	dlz_mysql_clear();
#endif

#ifdef DLZ_FILESYSTEM
        dlz_fs_clear();
#endif

#ifdef DLZ_BDB
        dlz_bdb_clear();
        dlz_bdbhpt_clear();
#endif

#ifdef DLZ_LDAP
        dlz_ldap_clear();
#endif

#ifdef DLZ_ODBC
        dlz_odbc_clear();
#endif

}
