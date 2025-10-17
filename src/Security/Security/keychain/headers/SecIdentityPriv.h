/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
    @header SecIdentityPriv
    The functions provided in SecIdentityPriv.h implement a convenient way to
    match private keys with certificates.
*/

#ifndef _SECURITY_SECIDENTITYPRIV_H_
#define _SECURITY_SECIDENTITYPRIV_H_

#include <Security/SecBase.h>
#include <Security/SecBasePriv.h>
#include <CoreFoundation/CoreFoundation.h>

__BEGIN_DECLS

/*! @function SecIdentityCreate
    @abstract create a new identity object from the provided certificate and its associated private key.
    @param allocator CFAllocator to allocate the identity object. Pass NULL to use the default allocator.
    @param certificate A certificate reference.
    @param privateKey A private key reference.
    @result An identity reference.
*/
SecIdentityRef SecIdentityCreate(
     CFAllocatorRef allocator,
     SecCertificateRef certificate,
     SecKeyRef privateKey)
    __SEC_MAC_AND_IOS_UNKNOWN;
    //__OSX_AVAILABLE_STARTING(__MAC_10_3, __SEC_IPHONE_UNKNOWN);

#if SEC_OS_OSX
/*!
    @function ConvertArrayToKeyUsage
    @abstract Given an array of key usages defined in SecItem.h return the equivalent CSSM_KEYUSE
    @param usage An CFArrayRef containing CFTypeRefs defined in SecItem.h
          kSecAttrCanEncrypt,
          kSecAttrCanDecrypt,
          kSecAttrCanDerive,
          kSecAttrCanSign,
          kSecAttrCanVerify,
          kSecAttrCanWrap,
          kSecAttrCanUnwrap
          If the CFArrayRef is NULL then the CSSM_KEYUSAGE will be CSSM_KEYUSE_ANY
    @result A CSSM_KEYUSE.  Derived from the passed in Array
*/
CSSM_KEYUSE ConvertArrayToKeyUsage(CFArrayRef usage)
  __SEC_MAC_ONLY_UNKNOWN;

/*!
    @function SecIdentityDeleteApplicationPreferenceItems
    @abstract Delete identity preference items created by the calling application.
    @result errSecSuccess on successful deletion, or errSecItemNotFound if no items
    were found to be deleted. Other keychain error results may be possible (SecBase.h).
    @discussion This function deletes all identity preference items which match the
    application identifier of the caller. This implies that items to be deleted were
    created with SecIdentitySetPreferred on a version of macOS where this function
    is implemented, since older versions of macOS did not add application identifier
    information. Note: currently, deletion is also limited to preference items whose
    name is in URI format.
*/
OSStatus SecIdentityDeleteApplicationPreferenceItems(void)
  __SEC_MAC_ONLY_UNKNOWN;
  //__OSX_AVAILABLE_STARTING(__MAC_11_3, __IPHONE_NA);

/*!
    @function SecIdentityCopyApplicationPreferenceItemURLs
    @abstract Returns an array of URLs that have corresponding identity preferences
    available to the calling application.
    @result An array of zero or more CFURLRefs for which identity preferences exist.
    Caller is responsible for releasing this array.
    @discussion This function returns an array of CFURLRef instances for all URL
    format identity preference items created by the caller. Given a URL in this array,
    your code can obtain the identity associated with that URL by calling
    SecIdentityCopyPreferred(CFURLGetString(url), NULL, NULL), or delete the preference
    by calling SecIdentitySetPreferred(NULL, CFURLGetString(url), NULL).
*/
CFArrayRef SecIdentityCopyApplicationPreferenceItemURLs(void)
  __SEC_MAC_ONLY_UNKNOWN;
  //__OSX_AVAILABLE_STARTING(__MAC_12_0, __IPHONE_NA);

#endif // SEC_OS_OSX

__END_DECLS

#endif /* _SECURITY_SECIDENTITYPRIV_H_ */
