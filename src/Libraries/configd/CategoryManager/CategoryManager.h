/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#ifndef _CATEGORY_MANAGER_H
#define _CATEGORY_MANAGER_H

/*
 * CategoryManager.h
 * - the CategoryManager API is a thin, stateless layer to handle just the IPC
 *   details
 */

/*
 * Modification History
 *
 * November 7, 2022	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#include <SystemConfiguration/SCNetworkCategoryManager.h>

/*
 * Block: CategoryManagerEventHandler
 * Purpose:
 *   Asynchronous event handler.
 *  
 * kCategoryManagerEventConnectionInterrupted
 *   Give the caller an opportunity to re-sync state with the server should
 *   the server crash/exit.
 *
 *   You can only invoke CategoryManagerConnectionSynchronize() within
 *   the reconnect callback. Calling the other API will deadlock unless
 *   dispatch_async'd onto a different queue.
 *
 * kCategoryManagerEventValueAcknowledged
 *   The active value has been acknowledged.
 */
typedef CF_ENUM(uint32_t, CategoryManagerEvent) {
	kCategoryManagerEventNone = 0,
	kCategoryManagerEventConnectionInvalid = 1,
	kCategoryManagerEventConnectionInterrupted = 2,
	kCategoryManagerEventValueAcknowledged = 3,
};

typedef void (^CategoryManagerEventHandler)(xpc_connection_t connection,
					    CategoryManagerEvent event);

xpc_connection_t
CategoryManagerConnectionCreate(dispatch_queue_t queue,
				CategoryManagerEventHandler handler);

errno_t
CategoryManagerConnectionRegister(xpc_connection_t connection,
				  CFStringRef category,
				  CFStringRef ifname,
				  SCNetworkCategoryManagerFlags flags);

errno_t
CategoryManagerConnectionActivateValue(xpc_connection_t connection,
				       CFStringRef value);

CFStringRef
CategoryManagerConnectionCopyActiveValue(xpc_connection_t connection,
					 int * error);
void
CategoryManagerConnectionSynchronize(xpc_connection_t connection,
				     CFStringRef category,
				     CFStringRef ifname,
				     SCNetworkCategoryManagerFlags flags,
				     CFStringRef value);

CFStringRef
CategoryManagerConnectionCopyActiveValueNoSession(xpc_connection_t connection,
						  CFStringRef category,
						  CFStringRef ifname,
						  int * error);

#endif /* _CATEGORY_MANAGER_H */
