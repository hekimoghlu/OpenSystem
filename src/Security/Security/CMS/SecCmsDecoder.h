/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
    @header SecCmsDecoder.h

    @availability 10.4 and later
    @abstract Interfaces of the CMS implementation.
    @discussion The functions here implement functions for encoding
                and decoding Cryptographic Message Syntax (CMS) objects
                as described in rfc3369.
 */

#ifndef _SECURITY_SECCMSDECODER_H_
#define _SECURITY_SECCMSDECODER_H_  1

#include <Security/SecCmsBase.h>

__BEGIN_DECLS

/*! @functiongroup Streaming interface */

#if TARGET_OS_OSX
/*!
     @function
     @abstract Set up decoding of a BER-encoded CMS message.
     @param arena An ArenaPool object to use for the resulting message, or NULL if new ArenaPool
     should be created.
     @param cb callback function for delivery of inner content inner
     content will be stored in the message if cb is NULL.
     @param cb_arg first argument passed to cb when it is called.
     @param pwfn callback function for getting token password for
     enveloped data content with a password recipient.
     @param pwfn_arg first argument passed to pwfn when it is called.
     @param decrypt_key_cb callback function for getting bulk key
     for encryptedData content.
     @param decrypt_key_cb_arg first argument passed to decrypt_key_cb
     when it is called.
     @param outDecoder On success will contain a pointer to a newly created SecCmsDecoder.
     @result A result code. See "SecCmsBase.h" for possible results.
     @discussion Create a SecCmsDecoder().  If this function returns noErr, the caller must dispose of the returned outDecoder by calling SecCmsDecoderDestroy() or SecCmsDecoderFinish().
     @availability 10.4 through 10.7
 */
extern OSStatus
SecCmsDecoderCreate(SecArenaPoolRef arena,
                    SecCmsContentCallback cb, void *cb_arg,
                    PK11PasswordFunc pwfn, void *pwfn_arg,
                    SecCmsGetDecryptKeyCallback decrypt_key_cb, void
                    *decrypt_key_cb_arg,
                    SecCmsDecoderRef *outDecoder)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);
#else // !TARGET_OS_OSX
/*!
    @function
    @abstract Set up decoding of a BER-encoded CMS message.
    @param cb callback function for delivery of inner content inner
	content will be stored in the message if cb is NULL.
    @param cb_arg first argument passed to cb when it is called.
    @param pwfn callback function for getting token password for
	enveloped data content with a password recipient.
    @param pwfn_arg first argument passed to pwfn when it is called.
    @param decrypt_key_cb callback function for getting bulk key
	for encryptedData content.
    @param decrypt_key_cb_arg first argument passed to decrypt_key_cb
	when it is called.
    @param outDecoder On success will contain a pointer to a newly created SecCmsDecoder.
    @result A result code. See "SecCmsBase.h" for possible results.
    @discussion Create a SecCmsDecoder().  If this function returns errSecSuccess, the caller must dispose of the returned outDecoder by calling SecCmsDecoderDestroy() or SecCmsDecoderFinish().
    @availability 10.4 and later
 */
extern OSStatus
SecCmsDecoderCreate(SecCmsContentCallback cb, void *cb_arg,
                   PK11PasswordFunc pwfn, void *pwfn_arg,
                   SecCmsGetDecryptKeyCallback decrypt_key_cb, void
                   *decrypt_key_cb_arg,
                   SecCmsDecoderRef *outDecoder)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#endif // !TARGET_OS_OSX

/*!
    @function
    @abstract Feed BER-encoded data to decoder.
    @param decoder Pointer to a SecCmsDecoderContext created with SecCmsDecoderCreate().
    @param buf Pointer to bytes to be decoded.
    @param len number of bytes to decode.
    @result A result code. See "SecCmsBase.h" for possible results.
    @discussion If a call to this function fails the caller should call SecCmsDecoderDestroy().
    @availability 10.4 and later
 */
extern OSStatus
SecCmsDecoderUpdate(SecCmsDecoderRef decoder, const void *buf, CFIndex len);

/*!
    @function
    @abstract Abort a (presumably failed) decoding process.
    @param decoder Pointer to a SecCmsDecoderContext created with SecCmsDecoderCreate().
    @availability 10.4 and later
 */
extern void
SecCmsDecoderDestroy(SecCmsDecoderRef decoder);

/*!
    @function
    @abstract Mark the end of inner content and finish decoding.
    @param decoder Pointer to a SecCmsDecoderContext created with SecCmsDecoderCreate().
    @param outMessage On success a pointer to a SecCmsMessage containing the decoded message.
    @result A result code. See "SecCmsBase.h" for possible results.
    @discussion decoder is no longer valid after this function is called.
    @availability 10.4 and later
 */
extern OSStatus
SecCmsDecoderFinish(SecCmsDecoderRef decoder, SecCmsMessageRef *outMessage);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

/*! @functiongroup One shot interface */
#if TARGET_OS_OSX
/*!
    @function
    @abstract Decode a CMS message from BER encoded data.
    @discussion This function basically does the same as calling
    SecCmsDecoderStart(), SecCmsDecoderUpdate() and SecCmsDecoderFinish().
    @param encodedMessage Pointer to a CSSM_DATA containing the BER encoded cms
    message to decode.
    @param cb callback function for delivery of inner content inner
    content will be stored in the message if cb is NULL.
    @param cb_arg first argument passed to cb when it is called.
    @param pwfn callback function for getting token password for enveloped
    data content with a password recipient.
    @param pwfn_arg first argument passed to pwfn when it is called.
    @param decrypt_key_cb callback function for getting bulk key for encryptedData content.
    @param decrypt_key_cb_arg first argument passed to decrypt_key_cb when it is called.
    @param outMessage On success a pointer to a SecCmsMessage containing the decoded message.
    @result A result code. See "SecCmsBase.h" for possible results.
    @discussion decoder is no longer valid after this function is called.
    @availability 10.4 through 10.7
 */
extern OSStatus
SecCmsMessageDecode(const CSSM_DATA *encodedMessage,
                    SecCmsContentCallback cb, void *cb_arg,
                    PK11PasswordFunc pwfn, void *pwfn_arg,
                    SecCmsGetDecryptKeyCallback decrypt_key_cb, void *decrypt_key_cb_arg,
                    SecCmsMessageRef *outMessage)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);
#else // !TARGET_OS_OSX
/*!
    @function
    @abstract Decode a CMS message from BER encoded data.
    @discussion This function basically does the same as calling
                SecCmsDecoderStart(), SecCmsDecoderUpdate() and SecCmsDecoderFinish().
    @param encodedMessage Pointer to a SecAsn1Item containing the BER encoded cms
           message to decode.
    @param cb callback function for delivery of inner content inner
           content will be stored in the message if cb is NULL.
    @param cb_arg first argument passed to cb when it is called.
    @param pwfn callback function for getting token password for enveloped
           data content with a password recipient.
    @param pwfn_arg first argument passed to pwfn when it is called.
    @param decrypt_key_cb callback function for getting bulk key for encryptedData content.
    @param decrypt_key_cb_arg first argument passed to decrypt_key_cb when it is called.
    @param outMessage On success a pointer to a SecCmsMessage containing the decoded message.
    @result A result code. See "SecCmsBase.h" for possible results.
    @discussion decoder is no longer valid after this function is called.
    @availability 10.4 and later
 */
extern OSStatus
SecCmsMessageDecode(const SecAsn1Item *encodedMessage,
                    SecCmsContentCallback cb, void *cb_arg,
                    PK11PasswordFunc pwfn, void *pwfn_arg,
                    SecCmsGetDecryptKeyCallback decrypt_key_cb, void *decrypt_key_cb_arg,
                    SecCmsMessageRef *outMessage)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#endif // !TARGET_OS_OSX

#pragma clang diagnostic pop


__END_DECLS

#endif /* _SECURITY_SECCMSDECODER_H_ */
