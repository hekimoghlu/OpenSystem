/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
/* $Id: dbtable.h,v 1.23 2007/06/19 23:47:16 tbox Exp $ */

#ifndef DNS_DBTABLE_H
#define DNS_DBTABLE_H 1

/*****
 ***** Module Info
 *****/

/*! \file dns/dbtable.h
 * \brief
 * DNS DB Tables
 *
 * XXX TBS XXX
 *
 * MP:
 *\li	The module ensures appropriate synchronization of data structures it
 *	creates and manipulates.
 *
 * Reliability:
 *\li	No anticipated impact.
 *
 * Resources:
 *\li	None.
 *
 * Security:
 *\li	No anticipated impact.
 *
 * Standards:
 *\li	None.
 */

#include <isc/lang.h>

#include <dns/types.h>

#define DNS_DBTABLEFIND_NOEXACT		0x01

ISC_LANG_BEGINDECLS

isc_result_t
dns_dbtable_create(isc_mem_t *mctx, dns_rdataclass_t rdclass,
		   dns_dbtable_t **dbtablep);
/*%<
 * Make a new dbtable of class 'rdclass'
 *
 * Requires:
 *\li	mctx != NULL
 * \li	dbtablep != NULL && *dptablep == NULL
 *\li	'rdclass' is a valid class
 *
 * Returns:
 *\li	#ISC_R_SUCCESS
 *\li	#ISC_R_NOMEMORY
 *\li	#ISC_R_UNEXPECTED
 */

void
dns_dbtable_attach(dns_dbtable_t *source, dns_dbtable_t **targetp);
/*%<
 * Attach '*targetp' to 'source'.
 *
 * Requires:
 *
 *\li	'source' is a valid dbtable.
 *
 *\li	'targetp' points to a NULL dns_dbtable_t *.
 *
 * Ensures:
 *
 *\li	*targetp is attached to source.
 */

void
dns_dbtable_detach(dns_dbtable_t **dbtablep);
/*%<
 * Detach *dbtablep from its dbtable.
 *
 * Requires:
 *
 *\li	'*dbtablep' points to a valid dbtable.
 *
 * Ensures:
 *
 *\li	*dbtablep is NULL.
 *
 *\li	If '*dbtablep' is the last reference to the dbtable,
 *		all resources used by the dbtable will be freed
 */

isc_result_t
dns_dbtable_add(dns_dbtable_t *dbtable, dns_db_t *db);
/*%<
 * Add 'db' to 'dbtable'.
 *
 * Requires:
 *\li	'dbtable' is a valid dbtable.
 *
 *\li	'db' is a valid database with the same class as 'dbtable'
 */

void
dns_dbtable_remove(dns_dbtable_t *dbtable, dns_db_t *db);
/*%<
 * Remove 'db' from 'dbtable'.
 *
 * Requires:
 *\li	'db' was previously added to 'dbtable'.
 */

void
dns_dbtable_adddefault(dns_dbtable_t *dbtable, dns_db_t *db);
/*%<
 * Use 'db' as the result of a dns_dbtable_find() if no better match is
 * available.
 */

void
dns_dbtable_getdefault(dns_dbtable_t *dbtable, dns_db_t **db);
/*%<
 * Get the 'db' used as the result of a dns_dbtable_find()
 * if no better match is available.
 */

void
dns_dbtable_removedefault(dns_dbtable_t *dbtable);
/*%<
 * Remove the default db from 'dbtable'.
 */

isc_result_t
dns_dbtable_find(dns_dbtable_t *dbtable, dns_name_t *name,
		 unsigned int options, dns_db_t **dbp);
/*%<
 * Find the deepest match to 'name' in the dbtable, and return it
 *
 * Notes:
 *\li	If the DNS_DBTABLEFIND_NOEXACT option is set, the best partial
 *	match (if any) to 'name' will be returned.
 *
 * Returns:  
 * \li #ISC_R_SUCCESS		on success
 *\li	     something else:		no default and match
 */

ISC_LANG_ENDDECLS

#endif /* DNS_DBTABLE_H */
