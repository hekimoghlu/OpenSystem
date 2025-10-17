/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#ifndef _LIBSYSTEMCONFIGURATION_SERVER_H
#define _LIBSYSTEMCONFIGURATION_SERVER_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <dispatch/dispatch.h>
#include <xpc/xpc.h>
#include <CoreFoundation/CoreFoundation.h>

// ------------------------------------------------------------

#pragma mark -
#pragma mark [XPC] information server common code

// ------------------------------------------------------------


/*
 * libSC_info_server_t
 *	data (CFData)
 *		stored configuration
 *	generation
 *		generation of the last stored configuration
 *	info (CFDictionary)
 *		key = xpc_connection_t [CFData]
 *		val = [CFData] (process name, last push'd, last ack'd)
 *	inSync_NO, inSync_YES
 *		count of client connections that have ack'd a configuration that
 *		are (or are not) in sync with the last stored generation#
*/
typedef struct {
	CFDataRef		data;
	uint64_t		generation;
	CFMutableDictionaryRef	info;
	int			inSync_NO;	// ack'ing clients not in sync w/generation
	int			inSync_YES;	// ack'ing clients in sync w/generation
} libSC_info_server_t;


__BEGIN_DECLS

// Global server info SPIs

void
_libSC_info_server_init		(
				 libSC_info_server_t	*server_info
				);

Boolean
_libSC_info_server_in_sync	(
				 libSC_info_server_t	*server_info
				);

void
_libSC_info_server_set_data	(
				 libSC_info_server_t	*server_info,
				 CFDataRef		data,
				 uint64_t		generation
				);

// Per-session server info SPIs

void
_libSC_info_server_open		(
				 libSC_info_server_t	*server_info,
				 xpc_connection_t	c
				);

CFDataRef
_libSC_info_server_get_data	(
				 libSC_info_server_t	*server_info,
				 xpc_connection_t	c,
				 uint64_t		*generation
				);

Boolean
_libSC_info_server_acknowledged	(
				 libSC_info_server_t	*server_info,
				 xpc_connection_t	c,
				 uint64_t		generation
				);

Boolean
_libSC_info_server_close	(
				 libSC_info_server_t	*server_info,
				 xpc_connection_t	c
				);

__END_DECLS

#endif	// _LIBSYSTEMCONFIGURATION_SERVER_H
