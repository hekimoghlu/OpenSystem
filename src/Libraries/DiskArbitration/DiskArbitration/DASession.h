/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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
#ifndef __DISKARBITRATION_DASESSION__
#define __DISKARBITRATION_DASESSION__

#include <CoreFoundation/CoreFoundation.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

#ifndef __DISKARBITRATIOND__

/*!
 * @typedef   DASessionRef
 * Type of a reference to DASession instances.
 */

typedef struct CF_BRIDGED_TYPE( id ) __DASession * DASessionRef API_AVAILABLE(macos(10.4));

/*!
 * @function   DASessionGetTypeID
 * @abstract   Returns the type identifier of all DASession instances.
 */

extern CFTypeID DASessionGetTypeID( void ) API_AVAILABLE(macos(10.4));

/*!
 * @function   DASessionCreate
 * @abstract   Creates a new session.
 * @result     A reference to a new DASession.
 * @discussion
 * The caller of this function receives a reference to the returned object.  The
 * caller also implicitly retains the object and is responsible for releasing it.
 */

extern DASessionRef __nullable DASessionCreate( CFAllocatorRef __nullable allocator ) API_AVAILABLE(macos(10.4));

/*!
 * @function   DASessionScheduleWithRunLoop
 * @abstract   Schedules the session on a run loop.
 * @param      session     The session which is being scheduled.
 * @param      runLoop     The run loop on which the session should be scheduled.
 * @param      runLoopMode The run loop mode in which the session should be scheduled.
 */

extern void DASessionScheduleWithRunLoop( DASessionRef session, CFRunLoopRef runLoop, CFStringRef runLoopMode ) API_AVAILABLE(macos(10.4));

/*!
 * @function   DASessionUnscheduleFromRunLoop
 * @abstract   Unschedules the session from a run loop.
 * @param      session     The session which is being unscheduled.
 * @param      runLoop     The run loop on which the session is scheduled.
 * @param      runLoopMode The run loop mode in which the session is scheduled.
 */

extern void DASessionUnscheduleFromRunLoop( DASessionRef session, CFRunLoopRef runLoop, CFStringRef runLoopMode ) API_AVAILABLE(macos(10.4));

/*!
 * @function   DASessionSetDispatchQueue
 * @abstract   Schedules the session on a dispatch queue.
 * @param      session The session which is being scheduled.
 * @param      queue   The dispatch queue on which the session should be scheduled.  Pass NULL to unschedule.
 */

extern void DASessionSetDispatchQueue( DASessionRef session, dispatch_queue_t __nullable queue ) API_AVAILABLE(macos(10.7));

/*
 * @typedef   DAApprovalSessionRef
 * Type of a reference to DAApprovalSession instances.
 */

typedef struct CF_BRIDGED_TYPE( id ) __DASession * DAApprovalSessionRef CF_SWIFT_UNAVAILABLE( "Use DASessionRef instead" );

/*
 * @function   DAApprovalSessionGetTypeID
 * @abstract   Returns the type identifier of all DAApprovalSession instances.
 */

extern CFTypeID DAApprovalSessionGetTypeID( void ) CF_SWIFT_UNAVAILABLE( "Use DASessionGetTypeID instead" );

/*
 * @function   DAApprovalSessionCreate
 * @abstract   Creates a new approval session.
 * @result     A reference to a new DAApprovalSession.
 * @discussion
 * The caller of this function receives a reference to the returned object.  The
 * caller also implicitly retains the object and is responsible for releasing it.
 */

extern DAApprovalSessionRef __nullable DAApprovalSessionCreate( CFAllocatorRef __nullable allocator ) CF_SWIFT_UNAVAILABLE( "Use DASessionCreate instead" ) API_AVAILABLE(macCatalyst(13.1)) API_UNAVAILABLE(ios, tvos, watchos);

/*
 * @function   DAApprovalSessionScheduleWithRunLoop
 * @abstract   Schedules the approval session on a run loop.
 * @param      session     The approval session which is being scheduled.
 * @param      runLoop     The run loop on which the approval session should be scheduled.
 * @param      runLoopMode The run loop mode in which the approval session should be scheduled.
 */

extern void DAApprovalSessionScheduleWithRunLoop( DAApprovalSessionRef session, CFRunLoopRef runLoop, CFStringRef runLoopMode ) CF_SWIFT_UNAVAILABLE( "Use DASessionSetDispatchQueue instead" ) API_AVAILABLE( macCatalyst(13.1)) API_UNAVAILABLE(ios, tvos, watchos);
/*
 * @function   DAApprovalSessionUnscheduleFromRunLoop
 * @abstract   Unschedules the approval session from a run loop.
 * @param      session     The approval session which is being unscheduled.
 * @param      runLoop     The run loop on which the approval session is scheduled.
 * @param      runLoopMode The run loop mode in which the approval session is scheduled.
 */

extern void DAApprovalSessionUnscheduleFromRunLoop( DAApprovalSessionRef session, CFRunLoopRef runLoop, CFStringRef runLoopMode ) CF_SWIFT_UNAVAILABLE( "Use DASessionSetDispatchQueue instead" ) API_AVAILABLE( macCatalyst(13.1)) API_UNAVAILABLE(ios, tvos, watchos);

#endif /* !__DISKARBITRATIOND__ */

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATION_DASESSION__ */
