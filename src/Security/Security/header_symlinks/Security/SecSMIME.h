/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
    @header SecSMIME.h

    @availability 10.4 and later
    @abstract S/MIME Specific routines.
    @discussion Header file for routines specific to S/MIME.  Keep
		things that are pure pkcs7 out of here; this is for
		S/MIME policy, S/MIME interoperability, etc.
*/

#ifndef _SECURITY_SECSMIME_H_
#define _SECURITY_SECSMIME_H_ 1

#include <Security/SecCmsBase.h>

__BEGIN_DECLS

/*!
    @function
    @abstract Find bulk algorithm suitable for all recipients.
 */
extern OSStatus
SecSMIMEFindBulkAlgForRecipients(SecCertificateRef *rcerts, SECOidTag *bulkalgtag, int *keysize);

__END_DECLS

#endif /* _SECURITY_SECSMIME_H_ */
