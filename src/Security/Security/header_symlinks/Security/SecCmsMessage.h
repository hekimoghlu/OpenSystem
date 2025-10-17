/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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
    @header SecCmsMessage.h

    @availability 10.4 and later
    @abstract CMS message object interfaces
    @abstract Interfaces of the CMS implementation.
    @discussion A SecCmsMessage represent a Cryptographic Message
                Syntax (CMS) object as described in rfc3369.
                It can be encoded using a SecCmsEncoder into BER
                data or obtained from a SecCmsDecoder and examined
                using the functions below.
 */

#ifndef _SECURITY_SECCMSMESSAGE_H_
#define _SECURITY_SECCMSMESSAGE_H_  1

#include <Security/SecCmsBase.h>

__BEGIN_DECLS

#if TARGET_OS_OSX

/*!
     @function
     @abstract Create a CMS message object.
     @param poolp Arena to allocate memory from, or NULL if new arena should
     be created.
     @result A pointer to a newly created SecCmsMessage.  When finished using
     this the caller should call SecCmsMessageDestroy().  On failure
     returns NULL.  In this case call PR_GetError() to find out what went
     wrong.
 */
extern SecCmsMessageRef
SecCmsMessageCreate(SecArenaPoolRef poolp)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);

#else // !TARGET_OS_OSX

/*!
    @function
    @abstract Create a CMS message object.
    @result A pointer to a newly created SecCmsMessage.  When finished using
        this the caller should call SecCmsMessageDestroy().  On failure
        returns NULL.  In this case call PR_GetError() to find out what went
        wrong.
 */
extern SecCmsMessageRef
SecCmsMessageCreate(void)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);

#endif // !TARGET_OS_OSX

/*!
    @function
    @abstract Destroy a CMS message and all of its sub-pieces.
    @param cmsg Pointer to a SecCmsMessage object.
 */
extern void
SecCmsMessageDestroy(SecCmsMessageRef cmsg);

/*!
    @function
    @abstract Return a copy of the given message.
    @discussion  The copy may be virtual or may be real -- either way, the
        result needs to be passed to SecCmsMessageDestroy later (as does the
        original).
    @param cmsg Pointer to a SecCmsMessage object.
 */
extern SecCmsMessageRef
SecCmsMessageCopy(SecCmsMessageRef cmsg);

#if SEC_OS_OSX
/*!
    @function
    @abstract Return a pointer to the message's arena pool.
 */
extern SecArenaPoolRef
SecCmsMessageGetArena(SecCmsMessageRef cmsg);
#endif

/*!
    @function
    @abstract Return a pointer to the top level contentInfo.
 */
extern SecCmsContentInfoRef
SecCmsMessageGetContentInfo(SecCmsMessageRef cmsg);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#if TARGET_OS_OSX
/*!
     @function
     @abstract Return a pointer to the actual content.
     @discussion In the case of those types which are encrypted, this returns the *plain* content.
     In case of nested contentInfos, this descends and retrieves the innermost content.
 */
extern CSSM_DATA_PTR
SecCmsMessageGetContent(SecCmsMessageRef cmsg)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);
#else // !TARGET_OS_OSX
/*!
    @function
    @abstract Return a pointer to the actual content. 
    @discussion In the case of those types which are encrypted, this returns the *plain* content.
                In case of nested contentInfos, this descends and retrieves the innermost content.
 */
extern const SecAsn1Item *
SecCmsMessageGetContent(SecCmsMessageRef cmsg)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#endif // !TARGET_OS_OSX

#pragma clang diagnostic pop

/*!
    @function
    @abstract Count number of levels of CMS content objects in this message.
    @discussion CMS data content objects do not count.
 */
extern int
SecCmsMessageContentLevelCount(SecCmsMessageRef cmsg);

/*!
    @function
    @abstract Find content level #n.
    @discussion CMS data content objects do not count.
 */
extern SecCmsContentInfoRef
SecCmsMessageContentLevel(SecCmsMessageRef cmsg, int n);

/*!
    @function
    @abstract See if message contains certs along the way.
 */
extern Boolean
SecCmsMessageContainsCertsOrCrls(SecCmsMessageRef cmsg);

/*!
    @function
    @abstract See if message contains a encrypted submessage.
 */
extern Boolean
SecCmsMessageIsEncrypted(SecCmsMessageRef cmsg);

/*!
    @function
    @abstract See if message contains a signed submessage
    @discussion If the CMS message has a SignedData with a signature (not just a SignedData)
                return true; false otherwise.  This can/should be called before calling
                VerifySignature, which will always indicate failure if no signature is
                present, but that does not mean there even was a signature!
                Note that the content itself can be empty (detached content was sent
                another way); it is the presence of the signature that matters.
 */
extern Boolean
SecCmsMessageIsSigned(SecCmsMessageRef cmsg);

/*!
    @function
    @abstract See if content is empty.
    @result Returns PR_TRUE is innermost content length is < minLen
    @discussion XXX need the encrypted content length (why?)
 */
extern Boolean
SecCmsMessageIsContentEmpty(SecCmsMessageRef cmsg, unsigned int minLen);

#if TARGET_OS_OSX
extern Boolean
SecCmsMessageContainsTSTInfo(SecCmsMessageRef cmsg);
#endif

__END_DECLS

#endif /* _SECURITY_SECCMSMESSAGE_H_ */
