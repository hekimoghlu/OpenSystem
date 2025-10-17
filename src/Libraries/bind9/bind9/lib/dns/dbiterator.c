/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
/* $Id: dbiterator.c,v 1.18 2007/06/19 23:47:16 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/util.h>

#include <dns/dbiterator.h>
#include <dns/name.h>

void
dns_dbiterator_destroy(dns_dbiterator_t **iteratorp) {
	/*
	 * Destroy '*iteratorp'.
	 */

	REQUIRE(iteratorp != NULL);
	REQUIRE(DNS_DBITERATOR_VALID(*iteratorp));

	(*iteratorp)->methods->destroy(iteratorp);

	ENSURE(*iteratorp == NULL);
}

isc_result_t
dns_dbiterator_first(dns_dbiterator_t *iterator) {
	/*
	 * Move the node cursor to the first node in the database (if any).
	 */

	REQUIRE(DNS_DBITERATOR_VALID(iterator));

	return (iterator->methods->first(iterator));
}

isc_result_t
dns_dbiterator_last(dns_dbiterator_t *iterator) {
	/*
	 * Move the node cursor to the first node in the database (if any).
	 */

	REQUIRE(DNS_DBITERATOR_VALID(iterator));

	return (iterator->methods->last(iterator));
}

isc_result_t
dns_dbiterator_seek(dns_dbiterator_t *iterator, dns_name_t *name) {
	/*
	 * Move the node cursor to the node with name 'name'.
	 */

	REQUIRE(DNS_DBITERATOR_VALID(iterator));

	return (iterator->methods->seek(iterator, name));
}

isc_result_t
dns_dbiterator_prev(dns_dbiterator_t *iterator) {
	/*
	 * Move the node cursor to the previous node in the database (if any).
	 */

	REQUIRE(DNS_DBITERATOR_VALID(iterator));

	return (iterator->methods->prev(iterator));
}

isc_result_t
dns_dbiterator_next(dns_dbiterator_t *iterator) {
	/*
	 * Move the node cursor to the next node in the database (if any).
	 */

	REQUIRE(DNS_DBITERATOR_VALID(iterator));

	return (iterator->methods->next(iterator));
}

isc_result_t
dns_dbiterator_current(dns_dbiterator_t *iterator, dns_dbnode_t **nodep,
		       dns_name_t *name)
{
	/*
	 * Return the current node.
	 */

	REQUIRE(DNS_DBITERATOR_VALID(iterator));
	REQUIRE(nodep != NULL && *nodep == NULL);
	REQUIRE(name == NULL || dns_name_hasbuffer(name));

	return (iterator->methods->current(iterator, nodep, name));
}

isc_result_t
dns_dbiterator_pause(dns_dbiterator_t *iterator) {
	/*
	 * Pause iteration.
	 */

	REQUIRE(DNS_DBITERATOR_VALID(iterator));

	return (iterator->methods->pause(iterator));
}

isc_result_t
dns_dbiterator_origin(dns_dbiterator_t *iterator, dns_name_t *name) {

	/*
	 * Return the origin to which returned node names are relative.
	 */

	REQUIRE(DNS_DBITERATOR_VALID(iterator));
	REQUIRE(iterator->relative_names);
	REQUIRE(dns_name_hasbuffer(name));

	return (iterator->methods->origin(iterator, name));
}

void
dns_dbiterator_setcleanmode(dns_dbiterator_t *iterator, isc_boolean_t mode) {
	REQUIRE(DNS_DBITERATOR_VALID(iterator));

	iterator->cleaning = mode;
}
