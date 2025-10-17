/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
/* $Id: rdatasetiter.c,v 1.16 2007/06/19 23:47:16 tbox Exp $ */

/*! \file */

#include <config.h>

#include <stddef.h>

#include <isc/util.h>

#include <dns/rdataset.h>
#include <dns/rdatasetiter.h>

void
dns_rdatasetiter_destroy(dns_rdatasetiter_t **iteratorp) {
	/*
	 * Destroy '*iteratorp'.
	 */

	REQUIRE(iteratorp != NULL);
	REQUIRE(DNS_RDATASETITER_VALID(*iteratorp));

	(*iteratorp)->methods->destroy(iteratorp);

	ENSURE(*iteratorp == NULL);
}

isc_result_t
dns_rdatasetiter_first(dns_rdatasetiter_t *iterator) {
	/*
	 * Move the rdataset cursor to the first rdataset at the node (if any).
	 */

	REQUIRE(DNS_RDATASETITER_VALID(iterator));

	return (iterator->methods->first(iterator));
}

isc_result_t
dns_rdatasetiter_next(dns_rdatasetiter_t *iterator) {
	/*
	 * Move the rdataset cursor to the next rdataset at the node (if any).
	 */

	REQUIRE(DNS_RDATASETITER_VALID(iterator));

	return (iterator->methods->next(iterator));
}

void
dns_rdatasetiter_current(dns_rdatasetiter_t *iterator,
			 dns_rdataset_t *rdataset)
{
	/*
	 * Return the current rdataset.
	 */

	REQUIRE(DNS_RDATASETITER_VALID(iterator));
	REQUIRE(DNS_RDATASET_VALID(rdataset));
	REQUIRE(! dns_rdataset_isassociated(rdataset));

	iterator->methods->current(iterator, rdataset);
}
