/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
    @header SecTrustStatusCodes
*/

#ifndef _SECURITY_SECTRUSTSTATUSCODES_H_
#define _SECURITY_SECTRUSTSTATUSCODES_H_

#include <Security/SecTrust.h>

__BEGIN_DECLS

/*!
 @function SecTrustCopyStatusCodes
 @abstract Returns a malloced array of SInt32 values, with the length in numStatusCodes,
 for the certificate specified by chain index in the given SecTrustRef.
 @param trust A reference to a trust object.
 @param index The index of the certificate whose status codes should be returned.
 @param numStatusCodes On return, the number of status codes allocated, or 0 if none.
 @result A pointer to an array of status codes, or NULL if no status codes exist.
 If the result is non-NULL, the caller must free() this pointer.
 @discussion This function returns an array of evaluation status codes for a certificate
 specified by its chain index in a trust reference. If NULL is returned, the certificate
 has no status codes.
 */
SInt32 *SecTrustCopyStatusCodes(SecTrustRef trust,
                                CFIndex index,
                                CFIndex *numStatusCodes);
__END_DECLS

#endif /* !_SECURITY_SECTRUSTSTATUSCODES_H_ */
