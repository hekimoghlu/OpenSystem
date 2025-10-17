/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <Availability.h>

/*
 * Type is any of the kGSSAttrTypeNNN credential types below, type are
 * strings
 */
extern const CFTypeRef kGSSAttrClass
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

extern const CFStringRef kGSSAttrClassKerberos
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFStringRef kGSSAttrClassNTLM
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFStringRef kGSSAttrClassIAKerb
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

/*
 * Item supports acquiring a gss_cred_id_t with GSSItemOperation
 */
extern const CFTypeRef kGSSAttrSupportGSSCredential
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

/*
 * kGSSAttrNameGSSExportedName, kGSSAttrNameGSSUsername,
 * kGSSAttrNameGSSServiceBasedHostname, can set and will be returned
 *
 * kGSSAttrNameDisplay can only be returned, constructed from the
 * other name types after creation.
 */
extern const CFTypeRef kGSSAttrNameType
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrNameTypeGSSExportedName /* CFDataRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrNameTypeGSSUsername /* CFStringRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrNameTypeGSSHostBasedService /* CFStringRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

extern const CFTypeRef kGSSAttrName
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

/* name suiteable to display to user */ 
extern const CFTypeRef kGSSAttrNameDisplay /* CFStringRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

/*
 * Unique UUID for this entry
 */
extern const CFTypeRef kGSSAttrUUID /* CFUUIDRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);


/*
 * If the item is a transient credential it can have associated
 * expiration time.
 */
extern const CFTypeRef kGSSAttrTransientExpire	/* CFDateRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrTransientDefaultInClass /* CFBooleanRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
/*
 * Credential to use to use when acquiring with with
 * GSSItemOperation(kGSSOperationAcquire) or when dealing with a
 * persistant credential.
 *
 * The credentials is not exportable and will always show up as
 * the cfobject kGSSAttrCredentialExists when queried.
 */

extern const CFTypeRef kGSSAttrCredentialPassword /* CFStringRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrCredentialStore /* CFBooleanRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrCredentialSecIdentity /* SecIdentityRef */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrCredentialExists
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

/*
 * Status of a credentials
 */

extern const CFTypeRef kGSSAttrStatusPersistant
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrStatusAutoAcquire
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrStatusAutoAcquireStatus
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSAttrStatusTransient
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

/*
 * Create/Modify/Delete/Search GSS items
 *
 * Credentials needs a type, name
 */

typedef struct GSSItem *GSSItemRef;

GSSItemRef
GSSItemAdd(CFDictionaryRef attributes, CFErrorRef *error)
	__attribute__((cf_returns_retained))
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

Boolean
GSSItemUpdate(CFDictionaryRef query, CFDictionaryRef attributesToUpdate, CFErrorRef *error)
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

Boolean
GSSItemDelete(CFDictionaryRef query, CFErrorRef *error)
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

Boolean
GSSItemDeleteItem(GSSItemRef item, CFErrorRef *error)
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

/**
 * Will never return a zero length array, GSSItemCopyMatching() will return more then one entry or a NULL pointer.
 */

CFArrayRef
GSSItemCopyMatching(CFDictionaryRef query, CFErrorRef *error)
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);



/*
 * Use a GSSItem to convert to either another type or to perform an
 * operation with the credential.
 *
 */

typedef struct __GSSOperationType const * GSSOperation;

extern const struct __GSSOperationType __kGSSOperationAcquire /* NULL, NULL|error */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
#define kGSSOperationAcquire (&__kGSSOperationAcquire)

extern const struct __GSSOperationType __kGSSOperationRenewCredential
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
#define kGSSOperationRenewCredential (&__kGSSOperationRenewCredential)

extern const struct __GSSOperationType __kGSSOperationGetGSSCredential /* gss_cred_it_t, NULL|error */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
#define kGSSOperationGetGSSCredential (&__kGSSOperationGetGSSCredential)

extern const struct __GSSOperationType __kGSSOperationDestoryTransient /* kCFBoolean{True,False}, NULL|error */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const struct __GSSOperationType __kGSSOperationDestroyTransient /* kCFBoolean{True,False}, NULL|error */
	__OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);
#define kGSSOperationDestoryTransient (&__kGSSOperationDestroyTransient)
#define kGSSOperationDestroyTransient (&__kGSSOperationDestroyTransient)

extern const struct __GSSOperationType __kGSSOperationRemoveBackingCredential /* kCFBoolean{True,False}, NULL|error */
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
#define kGSSOperationRemoveBackingCredential (&__kGSSOperationRemoveBackingCredential)

extern const struct __GSSOperationType __kGSSOperationChangePassword
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
#define kGSSOperationChangePassword (&__kGSSOperationChangePassword)

extern const CFTypeRef kGSSOperationChangePasswordOldPassword
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
extern const CFTypeRef kGSSOperationChangePasswordNewPassword
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

extern const struct __GSSOperationType __kGSSOperationCredentialDiagnostics
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
#define kGSSOperationCredentialDiagnostics (&__kGSSOperationCredentialDiagnostics)

extern const struct __GSSOperationType __kGSSOperationSetDefault
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
#define kGSSOperationSetDefault (&__kGSSOperationSetDefault)

typedef void (^GSSItemOperationCallbackBlock)(CFTypeRef result, CFErrorRef error);

Boolean
GSSItemOperation(GSSItemRef item, GSSOperation op, CFDictionaryRef options,
    dispatch_queue_t q, GSSItemOperationCallbackBlock fun)
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

CFTypeRef
GSSItemGetValue(GSSItemRef item, CFStringRef key)
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

CFTypeID
GSSItemGetTypeID(void)
	__OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);
