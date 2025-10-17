/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
/* $Id: bufferlist.h,v 1.17 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_BUFFERLIST_H
#define ISC_BUFFERLIST_H 1

/*****
 ***** Module Info
 *****/

/*! \file isc/bufferlist.h
 *
 *
 *\brief	Buffer lists have no synchronization.  Clients must ensure exclusive
 *	access.
 *
 * \li Reliability:
 *	No anticipated impact.

 * \li Security:
 *	No anticipated impact.
 *
 * \li Standards:
 *	None.
 */

/***
 *** Imports
 ***/

#include <isc/lang.h>
#include <isc/types.h>

ISC_LANG_BEGINDECLS

/***
 *** Functions
 ***/

unsigned int
isc_bufferlist_usedcount(isc_bufferlist_t *bl);
/*!<
 * \brief Return the length of the sum of all used regions of all buffers in
 * the buffer list 'bl'
 *
 * Requires:
 *
 *\li	'bl' is not NULL.
 *
 * Returns:
 *\li	sum of all used regions' lengths.
 */

unsigned int
isc_bufferlist_availablecount(isc_bufferlist_t *bl);
/*!<
 * \brief Return the length of the sum of all available regions of all buffers in
 * the buffer list 'bl'
 *
 * Requires:
 *
 *\li	'bl' is not NULL.
 *
 * Returns:
 *\li	sum of all available regions' lengths.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_BUFFERLIST_H */
