/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
/*!
 @header SecAccessControlPriv
 SecAccessControl defines access rights for items.
 */

#ifndef _SECURITY_SECACCESSCONTROLPRIV_H_
#define _SECURITY_SECACCESSCONTROLPRIV_H_

#include <Security/SecBase.h>
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFError.h>
#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFDictionary.h>

__BEGIN_DECLS

/*! Creates new empty access control object. */
SecAccessControlRef SecAccessControlCreate(CFAllocatorRef allocator, CFErrorRef *error);

// Protection, currently only kSecAttrAccessible* constants are allowed.  In future, another probable protection type might be CTK key object ID.
CFTypeRef SecAccessControlGetProtection(SecAccessControlRef access_control);
bool SecAccessControlSetProtection(SecAccessControlRef access_control, CFTypeRef protection, CFErrorRef *error);

/*! Represents constraint of the operation. */
typedef CFTypeRef SecAccessConstraintRef;

/*! Creates constraint based on specified policy.
    @param policy Identification of policy to be used.
 */
SecAccessConstraintRef SecAccessConstraintCreatePolicy(CFAllocatorRef allocator, CFTypeRef policy, CFErrorRef *error);

/*! Creates constraint which requires passcode verification. */
SecAccessConstraintRef SecAccessConstraintCreatePasscode(CFAllocatorRef allocator);

/*! Creates constraint which requires Touch ID or Face ID verification.*/
SecAccessConstraintRef SecAccessConstraintCreateBiometryAny(CFAllocatorRef allocator, CFDataRef catacombUUID);

/*! Creates constraint which requires Touch ID verification.*/
SecAccessConstraintRef SecAccessConstraintCreateTouchIDAny(CFAllocatorRef allocator, CFDataRef catacombUUID)
API_DEPRECATED_WITH_REPLACEMENT("SecAccessConstraintCreateBiometryAny", macos(10.12.1, 10.13.4), ios(9.0, 11.3));

/*! Creates constraint which requires Touch ID or Face ID verification.*/
SecAccessConstraintRef SecAccessConstraintCreateBiometryCurrentSet(CFAllocatorRef allocator, CFDataRef catacombUUID, CFDataRef bioDbHash);

/*! Creates constraint which requires Touch ID verification.*/
SecAccessConstraintRef SecAccessConstraintCreateTouchIDCurrentSet(CFAllocatorRef allocator, CFDataRef catacombUUID, CFDataRef bioDbHash)
API_DEPRECATED_WITH_REPLACEMENT("SecAccessConstraintCreateBiometryCurrentSet", macos(10.12.1, 10.13.4), ios(9.0, 11.3));

/*! Creates constraint which requires watch verification. */
SecAccessConstraintRef SecAccessConstraintCreateWatch(CFAllocatorRef allocator) API_DEPRECATED_WITH_REPLACEMENT("SecAccessConstraintCreateCompanion", macos(10.14, 15.0), ios(12.0, 18.0));

/*! Creates constraint which requires companion verification. */
SecAccessConstraintRef SecAccessConstraintCreateCompanion(CFAllocatorRef allocator) API_AVAILABLE(macos(15.0), ios(18.0));

/*! Creates constraint composed of other constraints.
    @param numRequired Number of constraints required to be satisfied in order to consider overal constraint satisfied.
    @param constraints Array of constraints to be chosen from.
 */
SecAccessConstraintRef SecAccessConstraintCreateKofN(CFAllocatorRef allocator, size_t numRequired, CFArrayRef constraints, CFErrorRef *error);

/*! Adds new constraint for specified operation.
    @param access_control Instance of access control object to add constraint to.
    @param operation Operation type.
    @param constraint Constraint object, created by one of SecAccessControlConstraintCreate() functions or kCFBooleanTrue
                      meaning that operation will be always allowed.
 */
bool SecAccessControlAddConstraintForOperation(SecAccessControlRef access_control, CFTypeRef operation,
                                               SecAccessConstraintRef constraint, CFErrorRef *error);

/*! Retrieves dictionary with constraint applicable for specified operation.
    @param access_control Instance of access control object to query.
    @param operation Operation type.
    @return Dictionary or kCFBooleanTrue representing constraint applied for requested operation.  If the operation
            is not allowed at all, NULL is returned.
 */
SecAccessConstraintRef SecAccessControlGetConstraint(SecAccessControlRef access_control, CFTypeRef operation);

/*! Serializes constraint applicable for specified operation into binary data form.
 @param access_control Instance of access control object to query.
 @param operation Operation type.
 @return Binary data representing constraint applied for requested operation
 */
CFDataRef SecAccessControlCopyConstraintData(SecAccessControlRef access_control, CFTypeRef operation);

/*! Retrieves dictionary with constraints keyed by operations (i.e. the ACL part of access control object).
    @return Dictionary with all constraints keyed by operation types.  Returns NULL if no operations are constrained.
 */
CFDictionaryRef SecAccessControlGetConstraints(SecAccessControlRef access_control);

/*! Sets dictionary with constraints for access control object.
 @param access_control Instance of access control object to set default access group to.
 @param constraints Constraint with all constraints.
 */
void SecAccessControlSetConstraints(SecAccessControlRef access_control, CFDictionaryRef constraints);

/*! Sets if application passwor is required.
@param require Indicate if password is required or not.
*/
void SecAccessControlSetRequirePassword(SecAccessControlRef access_control, bool require);

/*! Gets boolean value if application password is required.*/
bool SecAccessControlGetRequirePassword(SecAccessControlRef access_control);

/*! Sets if acl is bound.
 @param bound Indicate if password is bound or not.
 */
void SecAccessControlSetBound(SecAccessControlRef access_control, bool bound);

/*! Gets boolean value if acl is bound.*/
bool SecAccessControlIsBound(SecAccessControlRef access_control);

/*! Creates Access control instance from data serialized by SecAccessControlCopyData(). */
SecAccessControlRef SecAccessControlCreateFromData(CFAllocatorRef allocator, CFDataRef data, CFErrorRef *error);

/*! Serializes all access control object into binary data form. */
CFDataRef SecAccessControlCopyData(SecAccessControlRef access_control);

__END_DECLS

#endif // _SECURITY_SECACCESSCONTROLPRIV_H_
