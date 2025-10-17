/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
/*
 * Modification History
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * March 24, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _S_CONFIGD_SERVER_H
#define _S_CONFIGD_SERVER_H

#include <sys/cdefs.h>
#include <sys/fileport.h>
#include <mach/mach.h>
#include <CoreFoundation/CoreFoundation.h>

#define DISPATCH_MACH_SPI 1
#include <dispatch/private.h>

__BEGIN_DECLS

void			server_mach_channel_handler
					(void			*context,	// serverSessionRef
					 dispatch_mach_reason_t	reason,
					 dispatch_mach_msg_t	message,
					 mach_error_t		error);

void			server_init	(void);

dispatch_workloop_t	server_queue	(void);

kern_return_t	_snapshot	(mach_port_t		server,
				 int			*sc_status);

kern_return_t	_configopen	(mach_port_t		server,
				 xmlData_t		nameRef,
				 mach_msg_type_number_t	nameLen,
				 xmlData_t		optionsRef,
				 mach_msg_type_number_t	optionsLen,
				 mach_port_t		*newServer,
				 int			*sc_status,
				 audit_token_t		audit_token);

kern_return_t	_configlist	(mach_port_t server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 int			isRegex,
				 xmlDataOut_t		*listRef,
				 mach_msg_type_number_t	*listLen,
				 int			*sc_status);

kern_return_t	_configadd	(mach_port_t 		server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 xmlData_t		dataRef,
				 mach_msg_type_number_t	dataLen,
				 int			*newInstance,		// no longer used
				 int			*sc_status);


kern_return_t	_configadd_s	(mach_port_t 		server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 xmlData_t		dataRef,
				 mach_msg_type_number_t	dataLen,
				 int			*newInstance,		// no longer used
				 int			*sc_status);

kern_return_t	_configget	(mach_port_t		server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 xmlDataOut_t		*dataRef,
				 mach_msg_type_number_t	*dataLen,
				 int			*newInstance,		// no longer used
				 int			*sc_status);

kern_return_t	_configset	(mach_port_t		server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 xmlData_t		dataRef,
				 mach_msg_type_number_t	dataLen,
				 int			*newInstance,		// no longer used
				 int			*sc_status);

kern_return_t	_configremove	(mach_port_t		server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 int			*sc_status);

kern_return_t	_confignotify	(mach_port_t 		server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 int			*sc_status);

kern_return_t	_configget_m	(mach_port_t		server,
				 xmlData_t		keysRef,
				 mach_msg_type_number_t	keysLen,
				 xmlData_t		patternsRef,
				 mach_msg_type_number_t	patternsLen,
				 xmlDataOut_t		*dataRef,
				 mach_msg_type_number_t	*dataLen,
				 int			*sc_status);

kern_return_t	_configset_m	(mach_port_t		server,
				 xmlData_t		dataRef,
				 mach_msg_type_number_t	dataLen,
				 xmlData_t		removeRef,
				 mach_msg_type_number_t	removeLen,
				 xmlData_t		notifyRef,
				 mach_msg_type_number_t	notifyLen,
				 int			*sc_status);

kern_return_t	_notifyadd	(mach_port_t		server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 int			isRegex,
				 int			*status);

kern_return_t	_notifyremove	(mach_port_t		server,
				 xmlData_t		keyRef,
				 mach_msg_type_number_t	keyLen,
				 int			isRegex,
				 int			*status);

kern_return_t	_notifychanges	(mach_port_t		server,
				 xmlDataOut_t		*listRef,
				 mach_msg_type_number_t	*listLen,
				 int			*status);

kern_return_t	_notifyviaport	(mach_port_t		server,
				 mach_port_t		port,
				 mach_msg_id_t		msgid,
				 int			*status);

kern_return_t	_notifyviafd	(mach_port_t		server,
				 fileport_t		fileport,
				 int			identifier,
				 int			*status);

kern_return_t	_notifycancel	(mach_port_t		server,
				 int			*sc_status);

kern_return_t	_notifyset	(mach_port_t		server,
				 xmlData_t		keysRef,
				 mach_msg_type_number_t	keysLen,
				 xmlData_t		patternsRef,
				 mach_msg_type_number_t	patternsLen,
				 int			*sc_status);

__END_DECLS

#endif	/* !_S_CONFIGD_SERVER_H */
