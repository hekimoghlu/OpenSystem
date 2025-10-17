/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
/* $Id: resource.h,v 1.13 2008/07/11 23:47:09 tbox Exp $ */

#ifndef ISC_RESOURCE_H
#define ISC_RESOURCE_H 1

/*! \file isc/resource.h */

#include <isc/lang.h>
#include <isc/types.h>

#define ISC_RESOURCE_UNLIMITED ((isc_resourcevalue_t)ISC_UINT64_MAX)

ISC_LANG_BEGINDECLS

isc_result_t
isc_resource_setlimit(isc_resource_t resource, isc_resourcevalue_t value);
/*%<
 * Set the maximum limit for a system resource.
 *
 * Notes:
 *\li	If 'value' exceeds the maximum possible on the operating system,
 *	it is silently limited to that maximum -- or to "infinity", if
 *	the operating system has that concept.  #ISC_RESOURCE_UNLIMITED
 *	can be used to explicitly ask for the maximum.
 *
 * Requires:
 *\li	'resource' is a valid member of the isc_resource_t enumeration.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS	Success.
 *\li	#ISC_R_NOTIMPLEMENTED	'resource' is not a type known by the OS.
 *\li	#ISC_R_NOPERM	The calling process did not have adequate permission
 *			to change the resource limit.
 */

isc_result_t
isc_resource_getlimit(isc_resource_t resource, isc_resourcevalue_t *value);
/*%<
 * Get the maximum limit for a system resource.
 *
 * Notes:
 *\li	'value' is set to the maximum limit.
 *
 *\li	#ISC_RESOURCE_UNLIMITED is the maximum value of isc_resourcevalue_t.
 *
 *\li	On many (all?) Unix systems, RLIM_INFINITY is a valid value that is
 *	significantly less than #ISC_RESOURCE_UNLIMITED, but which in practice
 *	behaves the same.
 *
 *\li	The current ISC libdns configuration file parser assigns a value
 *	of ISC_UINT32_MAX for a size_spec of "unlimited" and ISC_UNIT32_MAX - 1
 *	for "default", the latter of which is supposed to represent "the
 *	limit that was in force when the server started".  Since these are
 *	valid values in the middle of the range of isc_resourcevalue_t,
 *	there is the possibility for confusion over what exactly those
 *	particular values are supposed to represent in a particular context --
 *	discrete integral values or generalized concepts.
 *
 * Requires:
 *\li	'resource' is a valid member of the isc_resource_t enumeration.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS		Success.
 *\li	#ISC_R_NOTIMPLEMENTED	'resource' is not a type known by the OS.
 */

isc_result_t
isc_resource_getcurlimit(isc_resource_t resource, isc_resourcevalue_t *value);
/*%<
 * Same as isc_resource_getlimit(), but returns the current (soft) limit.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS		Success.
 *\li	#ISC_R_NOTIMPLEMENTED	'resource' is not a type known by the OS.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_RESOURCE_H */

