/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#ifndef _LIBSYSTEMCONFIGURATION_CLIENT_H
#define _LIBSYSTEMCONFIGURATION_CLIENT_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <dispatch/dispatch.h>
#include <xpc/xpc.h>

// ------------------------------------------------------------

#pragma mark -
#pragma mark [XPC] DNS configuration server

#define	DNSINFO_SERVER_VERSION		20130408

#if	!TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
#define	DNSINFO_SERVICE_NAME		"com.apple.SystemConfiguration.DNSConfiguration"
#else	// !TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
#define	DNSINFO_SERVICE_NAME		"com.apple.SystemConfiguration.DNSConfiguration_sim"
#endif	// !TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST

#define	DNSINFO_PROC_NAME		"proc_name"	// string

#define	DNSINFO_REQUEST			"request_op"	// int64

enum {
	DNSINFO_REQUEST_COPY		= 0x10001,
	DNSINFO_REQUEST_ACKNOWLEDGE,
};

#define	DNSINFO_CONFIGURATION		"configuration"	// data
#define	DNSINFO_GENERATION		"generation"	// uint64

// ------------------------------------------------------------

#pragma mark -
#pragma mark [XPC] Network information (nwi) server

#define	NWI_SERVER_VERSION		20130408

#if	!TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
#define	NWI_SERVICE_NAME		"com.apple.SystemConfiguration.NetworkInformation"
#else	// !TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
#define	NWI_SERVICE_NAME		"com.apple.SystemConfiguration.NetworkInformation_sim"
#endif	// !TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST

#define	NWI_PROC_NAME			"proc_name"	// string

#define	NWI_REQUEST			"request_op"	// int64

enum {
	/* NWI state requests */
	NWI_STATE_REQUEST_COPY		= 0x20001,
	NWI_STATE_REQUEST_ACKNOWLEDGE,

#if	!TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
	/* NWI config agent requests  */
	NWI_CONFIG_AGENT_REQUEST_COPY
#endif	// !TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
};

#define	NWI_CONFIGURATION		"configuration"	// data
#define	NWI_GENERATION			"generation"	// uint64

// ------------------------------------------------------------

typedef struct {
	_Bool			active;
	xpc_connection_t	connection;
	char			*service_description;
	char			*service_name;
} libSC_info_client_t;

// ------------------------------------------------------------

__BEGIN_DECLS

_Bool
libSC_info_available			(void);

libSC_info_client_t *
libSC_info_client_create		(
					 dispatch_queue_t	q,
					 const char		*service_name,
					 const char		*service_description
					);

void
libSC_info_client_release		(
					 libSC_info_client_t	*client
					);

xpc_object_t
libSC_send_message_with_reply_sync	(
					 libSC_info_client_t	*client,
					 xpc_object_t		message
					);

__END_DECLS

#endif	// _LIBSYSTEMCONFIGURATION_CLIENT_H
