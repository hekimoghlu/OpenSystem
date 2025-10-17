/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
/* $Id: fsaccess.c,v 1.10 2007/06/19 23:47:17 tbox Exp $ */

/*! \file
 * \brief
 * This file contains the OS-independent functionality of the API.
 */
#include <isc/fsaccess.h>
#include <isc/result.h>
#include <isc/util.h>

/*!
 * Shorthand.  Maybe ISC__FSACCESS_PERMISSIONBITS should not even be in
 * <isc/fsaccess.h>.  Could check consistency with sizeof(isc_fsaccess_t)
 * and the number of bits in each function.
 */
#define STEP		(ISC__FSACCESS_PERMISSIONBITS)
#define GROUP		(STEP)
#define OTHER		(STEP * 2)

void
isc_fsaccess_add(int trustee, int permission, isc_fsaccess_t *access) {
	REQUIRE(trustee <= 0x7);
	REQUIRE(permission <= 0xFF);

	if ((trustee & ISC_FSACCESS_OWNER) != 0)
		*access |= permission;

	if ((trustee & ISC_FSACCESS_GROUP) != 0)
		*access |= (permission << GROUP);

	if ((trustee & ISC_FSACCESS_OTHER) != 0)
		*access |= (permission << OTHER);
}

void
isc_fsaccess_remove(int trustee, int permission, isc_fsaccess_t *access) {
	REQUIRE(trustee <= 0x7);
	REQUIRE(permission <= 0xFF);


	if ((trustee & ISC_FSACCESS_OWNER) != 0)
		*access &= ~permission;

	if ((trustee & ISC_FSACCESS_GROUP) != 0)
		*access &= ~(permission << GROUP);

	if ((trustee & ISC_FSACCESS_OTHER) != 0)
		*access &= ~(permission << OTHER);
}

static isc_result_t
check_bad_bits(isc_fsaccess_t access, isc_boolean_t is_dir) {
	isc_fsaccess_t bits;

	/*
	 * Check for disallowed user bits.
	 */
	if (is_dir)
		bits = ISC_FSACCESS_READ |
		       ISC_FSACCESS_WRITE |
		       ISC_FSACCESS_EXECUTE;
	else
		bits = ISC_FSACCESS_CREATECHILD |
		       ISC_FSACCESS_ACCESSCHILD |
		       ISC_FSACCESS_DELETECHILD |
		       ISC_FSACCESS_LISTDIRECTORY;

	/*
	 * Set group bad bits.
	 */
	bits |= bits << STEP;
	/*
	 * Set other bad bits.
	 */
	bits |= bits << STEP;

	if ((access & bits) != 0) {
		if (is_dir)
			return (ISC_R_NOTFILE);
		else
			return (ISC_R_NOTDIRECTORY);
	}

	return (ISC_R_SUCCESS);
}
