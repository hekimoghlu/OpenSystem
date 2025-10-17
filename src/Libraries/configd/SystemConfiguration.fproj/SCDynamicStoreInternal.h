/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#ifndef _SCDYNAMICSTOREINTERNAL_H
#define _SCDYNAMICSTOREINTERNAL_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <dispatch/dispatch.h>
#include <sys/types.h>
#include <mach/mach.h>
#include <pthread.h>
#include <regex.h>
#include <os/log.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFRuntime.h>

#ifndef	SC_LOG_HANDLE
#define	SC_LOG_HANDLE	__log_SCDynamicStore
#endif	// SC_LOG_HANDLE
#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCPrivate.h>
#include <SystemConfiguration/SCValidation.h>

/* Define the status of any registered notification. */
typedef enum {
	NotifierNotRegistered = 0,
	Using_NotifierWait,
	Using_NotifierInformViaMachPort,
	Using_NotifierInformViaFD,
	Using_NotifierInformViaRunLoop,
	Using_NotifierInformViaDispatch
} __SCDynamicStoreNotificationStatus;


typedef struct __SCDynamicStore {

	/* base CFType information */
	CFRuntimeBase			cfBase;

	/* queue (protects storePrivate->server) */
	dispatch_queue_t		queue;

	/* client side of the "configd" session */
	CFStringRef			name;
	CFDictionaryRef			options;

	/* server side of the "configd" session */
	mach_port_t			server;		// [dispatch] sync to storePrivate->lock

	/* per-session flags */
	Boolean				useSessionKeys;

	/* current status of notification requests */
	__SCDynamicStoreNotificationStatus	notifyStatus;

	/* "client" information associated with SCDynamicStoreCreateRunLoopSource() */
	CFMutableArrayRef		rlList;
	CFRunLoopSourceRef		rls;
	SCDynamicStoreCallBack		rlsFunction;
	SCDynamicStoreContext		rlsContext;
	CFMachPortRef			rlsNotifyPort;
	CFRunLoopSourceRef		rlsNotifyRLS;

	/* "client" information associated with SCDynamicStoreSetDispatchQueue() */
	dispatch_queue_t		dispatchQueue;
	dispatch_source_t		dispatchSource;

	/* "client" information associated with SCDynamicStoreSetDisconnectCallBack() */
	SCDynamicStoreDisconnectCallBack	disconnectFunction;
	Boolean					disconnectForceCallBack;

	/* SCDynamicStoreKeys being watched */
	CFMutableArrayRef		keys;
	CFMutableArrayRef		patterns;

	/* "server" information associated with mach port based notifications */
	mach_port_t			notifyPort;
	mach_msg_id_t			notifyPortIdentifier;

	/* "server" information associated with SCDynamicStoreNotifyFileDescriptor() */
	int				notifyFile;
	int				notifyFileIdentifier;

	/* "client" caching */
	Boolean				cache_active;
	CFMutableDictionaryRef		cached_keys;
	CFMutableDictionaryRef		cached_set;
	CFMutableArrayRef		cached_removals;
	CFMutableArrayRef		cached_notifys;

} SCDynamicStorePrivate, *SCDynamicStorePrivateRef;


__BEGIN_DECLS

static __inline__ CFTypeRef
isA_SCDynamicStore(CFTypeRef obj)
{
	return (isA_CFType(obj, SCDynamicStoreGetTypeID()));
}

__private_extern__
os_log_t
__log_SCDynamicStore			(void);

SCDynamicStorePrivateRef
__SCDynamicStoreCreatePrivate		(CFAllocatorRef			allocator,
					 const CFStringRef		name,
					 SCDynamicStoreCallBack		callout,
					 SCDynamicStoreContext		*context);

__private_extern__
Boolean
__SCDynamicStoreNormalize		(SCDynamicStoreRef		*store,
					 Boolean			allowNullSession);

__private_extern__
mach_port_t
__SCDynamicStoreAddNotificationPort	(SCDynamicStoreRef		store);

__private_extern__
void
__SCDynamicStoreRemoveNotificationPort	(SCDynamicStoreRef		store,
					 mach_port_t			port);

__private_extern__
Boolean
__SCDynamicStoreCheckRetryAndHandleError(SCDynamicStoreRef		store,
					 kern_return_t			status,
					 int				*sc_status,
					 const char			*func);

__private_extern__
Boolean
__SCDynamicStoreReconnectNotifications	(SCDynamicStoreRef		store);

__private_extern__
int
__SCDynamicStoreMapInternalStatus	(int				sc_status,
					 Boolean			generate_fault);

__private_extern__
CFPropertyListRef
__SCDynamicStoreCopyValueCommon		(SCDynamicStoreRef 		store,
					 CFStringRef 			key,
					 Boolean 			preserve_status);

__END_DECLS

#endif	/* _SCDYNAMICSTOREINTERNAL_H */
