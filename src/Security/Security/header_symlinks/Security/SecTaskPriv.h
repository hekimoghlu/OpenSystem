/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#ifndef _SECURITY_SECTASKPRIV_H_
#define _SECURITY_SECTASKPRIV_H_

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecTask.h>
#include <xpc/xpc.h>

__BEGIN_DECLS

#if SEC_OS_OSX
/*!
    @function SecTaskValidateForRequirement
    @abstract Validate a SecTask instance for a specified requirement.
    @param task The SecTask instance to validate.
    @param requirement A requirement string to be validated.
    @result An error code of type OSStatus. Returns errSecSuccess if the
    task satisfies the requirement.
*/

OSStatus SecTaskValidateForRequirement(SecTaskRef _Nonnull task, CFStringRef _Nonnull requirement);

#endif /* SEC_OS_OSX */

/*!
 @function SecTaskCreateWithXPCMessage
 @abstract Get SecTask instance from the remote peer of the xpc connection
 @param message message from peer in the xpc connection event handler, you can't use
                connection since it cached and uses the most recent sender to this connection.
 */

_Nullable SecTaskRef
SecTaskCreateWithXPCMessage(xpc_object_t _Nonnull message);

/*!
 @function SecTaskEntitlementsValidated
 @abstract Check whether entitlements can be trusted or not.  If this returns
 false the tasks entitlements must not be used for anything security sensetive.
 @param task A previously created SecTask object
 */
Boolean SecTaskEntitlementsValidated(SecTaskRef _Nonnull task);

/*!
 @function SecTaskCopyTeamIdentifier
 @abstract Return the value of the team identifier.
 @param task A previously created SecTask object
 @param error On a NULL return, this will contain a CFError describing
 the problem.  This argument may be NULL if the caller is not interested in
 detailed errors. The caller must CFRelease the returned value
 */
__nullable
CFStringRef SecTaskCopyTeamIdentifier(SecTaskRef _Nonnull task, CFErrorRef _Nullable * _Nullable error);

#if !TARGET_OS_SIMULATOR
CF_IMPLICIT_BRIDGING_ENABLED
/*!
 @function SecTaskValidateForLightweightCodeRequirementData
 @abstract Match the DER encoded requirement against the running task.
 @param task A previously created SecTask object
 @param requirement The DER encoded requirement to match against.
 @param error A CFError set if the requirement does not match the task.
 @result True if there requirement matches the task and false otherwise.
 */
bool SecTaskValidateForLightweightCodeRequirementData(SecTaskRef _Nonnull task, CFDataRef _Nonnull requirement, CFErrorRef _Nullable * _Nullable error);
CF_IMPLICIT_BRIDGING_DISABLED
#endif

__END_DECLS

#endif /* !_SECURITY_SECTASKPRIV_H_ */
