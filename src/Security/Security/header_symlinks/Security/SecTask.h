/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#ifndef _SECURITY_SECTASK_H_
#define _SECURITY_SECTASK_H_

#include <Security/SecBase.h>

#include <CoreFoundation/CoreFoundation.h>
#include <mach/message.h>

#include <sys/cdefs.h>

#if SEC_OS_OSX
#include <Security/SecCode.h>
#endif /* SEC_OS_OSX */

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

/*!
    @typedef SecTaskRef
    @abstract CFType used for representing a task
*/
typedef struct CF_BRIDGED_TYPE(id) __SecTask *SecTaskRef;

/*!
    @function SecTaskGetTypeID
    @abstract Returns the type ID for CF instances of SecTask.
    @result A CFTypeID for SecTask
*/
CFTypeID SecTaskGetTypeID(void);

/*!
    @function SecTaskCreateWithAuditToken
    @abstract Create a SecTask object for the task that sent the mach message
    represented by the audit token.
    @param token The audit token of a mach message
    @result The newly created SecTask object or NULL on error.  The caller must
    CFRelease the returned object.
*/
__nullable
SecTaskRef SecTaskCreateWithAuditToken(CFAllocatorRef __nullable allocator, audit_token_t token);

/*!
    @function SecTaskCreateFromSelf
    @abstract Create a SecTask object for the current task.
    @result The newly created SecTask object or NULL on error.  The caller must
    CFRelease the returned object.
#ifndef LEFT
*/
__nullable
SecTaskRef SecTaskCreateFromSelf(CFAllocatorRef __nullable allocator);

/*!
    @function SecTaskCopyValueForEntitlement
    @abstract Returns the value of a single entitlement for the represented 
    task.
    @param task A previously created SecTask object
    @param entitlement The name of the entitlement to be fetched
    @param error On a NULL return, this may be contain a CFError describing
    the problem.  This argument may be NULL if the caller is not interested in
    detailed errors.
    @result The value of the specified entitlement for the process or NULL if
    the entitlement value could not be retrieved.  The type of the returned
    value will depend on the entitlement specified.  The caller must release
    the returned object.
    @discussion A NULL return may indicate an error, or it may indicate that
    the entitlement is simply not present.  In the latter case, no CFError is
    returned.
*/
__nullable
CFTypeRef SecTaskCopyValueForEntitlement(SecTaskRef task, CFStringRef entitlement, CFErrorRef *error);

/*!
    @function SecTaskCopyValuesForEntitlements
    @abstract Returns the values of multiple entitlements for the represented 
    task.
    @param task A previously created SecTask object
    @param entitlements An array of entitlement names to be fetched
    @param error On a NULL return, this will contain a CFError describing
    the problem.  This argument may be NULL if the caller is not interested in
    detailed errors.  If a requested entitlement is not present for the 
    returned dictionary, the entitlement is not set on the task.  The caller
    must CFRelease the returned value
*/
__nullable
CFDictionaryRef SecTaskCopyValuesForEntitlements(SecTaskRef task, CFArrayRef entitlements, CFErrorRef *error);

/*!
    @function SecTaskCopySigningIdentifier
    @abstract Return the value of the codesigning identifier.
    @param task A previously created SecTask object
    @param error On a NULL return, this will contain a CFError describing
    the problem.  This argument may be NULL if the caller is not interested in
    detailed errors. The caller must CFRelease the returned value
*/
__nullable
CFStringRef SecTaskCopySigningIdentifier(SecTaskRef task, CFErrorRef *error);

/*!
    @function SecTaskGetCodeSignStatus
    @abstract Return the code sign status flags
    @param task A previously created SecTask object
*/

uint32_t SecTaskGetCodeSignStatus(SecTaskRef task)
    API_AVAILABLE(ios(10.0), watchos(3.0), tvos(10.0), macCatalyst(11.0)) SPI_AVAILABLE(macos(10.5));


CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS

#endif /* !_SECURITY_SECTASK_H_ */
