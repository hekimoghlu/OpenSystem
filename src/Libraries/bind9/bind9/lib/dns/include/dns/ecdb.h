/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
/* $Id: ecdb.h,v 1.3 2009/09/02 23:48:02 tbox Exp $ */

#ifndef DNS_ECDB_H
#define DNS_ECDB_H 1

/*****
 ***** Module Info
 *****/

/* TBD */

/***
 *** Imports
 ***/

#include <dns/types.h>

/***
 *** Types
 ***/

/***
 *** Functions
 ***/

ISC_LANG_BEGINDECLS

/* TBD: describe those */

isc_result_t
dns_ecdb_register(isc_mem_t *mctx, dns_dbimplementation_t **dbimp);

void
dns_ecdb_unregister(dns_dbimplementation_t **dbimp);

ISC_LANG_ENDDECLS

#endif /* DNS_ECDB_H */
