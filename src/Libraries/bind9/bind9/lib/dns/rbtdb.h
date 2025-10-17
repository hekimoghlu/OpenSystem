/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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

#ifndef DNS_RBTDB_H
#define DNS_RBTDB_H 1

#include <isc/lang.h>
#include <dns/types.h>

/*****
 ***** Module Info
 *****/

/*! \file
 * \brief
 * DNS Red-Black Tree DB Implementation
 */

ISC_LANG_BEGINDECLS

isc_result_t
dns_rbtdb_create(isc_mem_t *mctx, dns_name_t *base, dns_dbtype_t type,
		 dns_rdataclass_t rdclass, unsigned int argc, char *argv[],
		 void *driverarg, dns_db_t **dbp);

/*%<
 * Create a new database of type "rbt" (or "rbt64").  Called via
 * dns_db_create(); see documentation for that function for more details.
 *
 * If argv[0] is set, it points to a valid memory context to be used for
 * allocation of heap memory.  Generally this is used for cache databases
 * only.
 *
 * Requires:
 *
 * \li argc == 0 or argv[0] is a valid memory context.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_RBTDB_H */
