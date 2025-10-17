/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */

#ifndef __OS_OSMESSAGENOTIFICATION_H
#define __OS_OSMESSAGENOTIFICATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <mach/mach_types.h>
#include <device/device_types.h>
#include <IOKit/IOReturn.h>

enum {
	kFirstIOKitNotificationType                 = 100,
	kIOServicePublishNotificationType           = 100,
	kIOServiceMatchedNotificationType           = 101,
	kIOServiceTerminatedNotificationType        = 102,
	kIOAsyncCompletionNotificationType          = 150,
	kIOServiceMessageNotificationType           = 160,
	kLastIOKitNotificationType                  = 199,

	// reserved bits
	kIOKitNoticationTypeMask                    = 0x00000FFF,
	kIOKitNoticationTypeSizeAdjShift            = 30,
	kIOKitNoticationMsgSizeMask                 = 3,
};

enum {
	kOSNotificationMessageID            = 53,
	kOSAsyncCompleteMessageID           = 57,
	kMaxAsyncArgs                       = 16
};

enum {
	kIOAsyncReservedIndex       = 0,
	kIOAsyncReservedCount,

	kIOAsyncCalloutFuncIndex    = kIOAsyncReservedCount,
	kIOAsyncCalloutRefconIndex,
	kIOAsyncCalloutCount,

	kIOMatchingCalloutFuncIndex = kIOAsyncReservedCount,
	kIOMatchingCalloutRefconIndex,
	kIOMatchingCalloutCount,

	kIOInterestCalloutFuncIndex = kIOAsyncReservedCount,
	kIOInterestCalloutRefconIndex,
	kIOInterestCalloutServiceIndex,
	kIOInterestCalloutCount
};



// --------------
enum {
	kOSAsyncRef64Count  = 8,
	kOSAsyncRef64Size   = kOSAsyncRef64Count * ((int) sizeof(io_user_reference_t))
};
typedef io_user_reference_t OSAsyncReference64[kOSAsyncRef64Count];

struct OSNotificationHeader64 {
	mach_msg_size_t     size;       /* content size */
	natural_t           type;
	OSAsyncReference64  reference;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
	unsigned char       content[];
#else
	unsigned char       content[0];
#endif
};

#pragma pack(4)
struct IOServiceInterestContent64 {
	natural_t           messageType;
	io_user_reference_t messageArgument[1];
};
#pragma pack()
// --------------

#if !KERNEL_USER32

enum {
	kOSAsyncRefCount    = 8,
	kOSAsyncRefSize     = 32
};
typedef natural_t OSAsyncReference[kOSAsyncRefCount] __kernel_ptr_semantics;

struct OSNotificationHeader {
	mach_msg_size_t     size;       /* content size */
	natural_t           type;
	OSAsyncReference    reference;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
	unsigned char       content[];
#else
	unsigned char       content[0];
#endif
};

#pragma pack(4)
struct IOServiceInterestContent {
	natural_t   messageType;
	void *      messageArgument[1];
};
#pragma pack()

#endif /* KERNEL_USER32  */

struct IOAsyncCompletionContent {
	IOReturn result;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
	void * args[] __attribute__ ((packed));
#else
	void * args[0] __attribute__ ((packed));
#endif
};

#ifndef __cplusplus
typedef struct OSNotificationHeader OSNotificationHeader;
typedef struct IOServiceInterestContent IOServiceInterestContent;
typedef struct IOAsyncCompletionContent IOAsyncCompletionContent;
#endif

#ifdef __cplusplus
}
#endif

#endif /*  __OS_OSMESSAGENOTIFICATION_H */
