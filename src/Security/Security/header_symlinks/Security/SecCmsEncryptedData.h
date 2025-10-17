/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
    @header SecCmsEnvelopedData.h

    @availability 10.4 and later
    @abstract Interfaces of the CMS implementation.
    @discussion The functions here implement functions for encoding
                and decoding Cryptographic Message Syntax (CMS) objects
                as described in rfc3369.
 */

#ifndef _SECURITY_SECCMSENCRYPTEDDATA_H_
#define _SECURITY_SECCMSENCRYPTEDDATA_H_  1

#include <Security/SecCmsBase.h>

__BEGIN_DECLS

/*!
    @function
    @abstract Create an empty EncryptedData object.
    @param algorithm Specifies the bulk encryption algorithm to use.
    @param keysize is the key size.
    @discussion An error results in a return value of NULL and an error set.
                (Retrieve specific errors via PORT_GetError()/XP_GetError().)
 */
extern SecCmsEncryptedDataRef
SecCmsEncryptedDataCreate(SecCmsMessageRef cmsg, SECOidTag algorithm, int keysize);

/*!
    @function
    @abstract Destroy an encryptedData object.
 */
extern void
SecCmsEncryptedDataDestroy(SecCmsEncryptedDataRef encd);

/*!
    @function
    @abstract Return pointer to an EncryptedData object's contentInfo.
 */
extern SecCmsContentInfoRef
SecCmsEncryptedDataGetContentInfo(SecCmsEncryptedDataRef encd);

__END_DECLS

#endif /* _SECURITY_SECCMSENCRYPTEDDATA_H_ */
