/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
/* $Id: rootns.h,v 1.16 2007/06/19 23:47:17 tbox Exp $ */

#ifndef DNS_ROOTNS_H
#define DNS_ROOTNS_H 1

/*! \file dns/rootns.h */

#include <isc/lang.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
dns_rootns_create(isc_mem_t *mctx, dns_rdataclass_t rdclass,
                  const char *filename, dns_db_t **target);

void
dns_root_checkhints(dns_view_t *view, dns_db_t *hints, dns_db_t *db);
/*
 * Reports differences between hints and the real roots.
 *
 * Requires view, hints and (cache) db to be valid.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_ROOTNS_H */
