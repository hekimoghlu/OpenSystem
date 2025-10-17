/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
@header SecPassword
	SecPassword implements logic to use the system facilities for acquiring a password,
    optionally stored and retrieved from the user's keychain.
 */

#include <Security/SecBase.h>
#include <Security/SecKeychainItem.h>
#include <Security/cssmapple.h>

#ifndef _SECURITY_SECPASSWORD_H_
#define _SECURITY_SECPASSWORD_H_

#if defined(__cplusplus)
extern "C" {
#endif

/*!
    @abstract Flags to specify SecPasswordAction behavior, as the application steps through the options
    Get, just get it.
    Get|Set, get it and set it if it wasn't in the keychain; client doesn't verify it before it's stored
    Get|Fail, get it and flag that the previously given or stored password is busted.
    Get|Set|Fail, same as above but also store it.
    New instead of Get toggles between asking for a new passphrase and an existing one.
*/
enum {
    kSecPasswordGet     = 1<<0, // Get password from keychain or user
    kSecPasswordSet     = 1<<1, // Set password (passed in if kSecPasswordGet not set, otherwise from user)
    kSecPasswordFail    = 1<<2,  // Wrong password (ignore item in keychain and flag error)
    kSecPasswordNew     = 1<<3  // Explicitly get a new passphrase
};

/*!
    @function SecGenericPasswordCreate
    @abstract Create an SecPassword object be used with SecPasswordAction to query and/or set a password used in the client.
			The keychain list is searched for a generic password with the supplied attributes.  If
			the item is not found, SecPasswordAction will create a new password in the default keychain.
			Otherwise, the existing item is updated.
			searchAttrList and itemAttrList are optional - pass NULL for both of them if you only wish to query the user for a password.
            Use CFRelease on the returned SecPasswordRef when it is no longer needed.
    @param searchAttrList (in/opt) The list of search attributes for the item.
	@param itemAttrList (in/opt) A list of attributes which will be used for item creation.
    @param itemRef (out) On return, a pointer to a password reference.  Release this by calling the CFRelease function.
 */
OSStatus SecGenericPasswordCreate(SecKeychainAttributeList *searchAttrList, SecKeychainAttributeList *itemAttrList, SecPasswordRef *itemRef)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
    @function SecPasswordAction
    @abstract Get the password for a SecPassword, either from the user or the keychain and return it.
    Use SecKeychainItemFreeContent to free the data.

	@param itemRef An itemRef previously obtained from SecGenericPasswordCreate.
    @param message Message to display to the user as a CFString or nil for a default message.
        (future extension accepts CFDictionary for other hints, icon, secaccess)
    @param flags (in) The mode of operation.  See the flags documentation above.
    @param length (out) The length of the buffer pointed to by data.
	@param data A pointer to a buffer containing the data to store.

 */
OSStatus SecPasswordAction(SecPasswordRef itemRef, CFTypeRef message, UInt32 flags, UInt32 *length, const void **data)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
    @function SecPasswordSetInitialAccess
    @abstract Set the initial access ref.  Only used when a password is first added to the keychain.
 */
OSStatus SecPasswordSetInitialAccess(SecPasswordRef itemRef, SecAccessRef accessRef)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_SECPASSWORD_H_ */
