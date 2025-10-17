/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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
/* $Id: netscope.h,v 1.13 2009/06/25 23:48:02 tbox Exp $ */

#ifndef ISC_NETSCOPE_H
#define ISC_NETSCOPE_H 1

/*! \file isc/netscope.h */

ISC_LANG_BEGINDECLS

/*%
 * Convert a string of an IPv6 scope zone to zone index.  If the conversion
 * succeeds, 'zoneid' will store the index value.
 *
 * XXXJT: when a standard interface for this purpose is defined,
 * we should use it.
 *
 * Returns:
 * \li	ISC_R_SUCCESS: conversion succeeds
 * \li	ISC_R_FAILURE: conversion fails
 */
isc_result_t
isc_netscope_pton(int af, char *scopename, void *addr, isc_uint32_t *zoneid);

ISC_LANG_ENDDECLS

#endif /* ISC_NETSCOPE_H */
