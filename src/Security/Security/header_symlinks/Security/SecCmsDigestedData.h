/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
    @header SecCmsDigestedData.h

    @availability 10.4 and later
    @abstract Interfaces of the CMS implementation.
    @discussion The functions here implement functions for creating
                and accessing the DigestData content type of a 
                Cryptographic Message Syntax (CMS) object
                as described in rfc3369.
 */

#ifndef _SECURITY_SECCMSDIGESTEDDATA_H_
#define _SECURITY_SECCMSDIGESTEDDATA_H_  1

#include <Security/SecCmsBase.h>

__BEGIN_DECLS

/*!
    @function
    @abstract Create a digestedData object (presumably for encoding).
    @discussion Version will be set by SecCmsDigestedDataEncodeBeforeStart
                digestAlg is passed as parameter
                contentInfo must be filled by the user
                digest will be calculated while encoding
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
extern SecCmsDigestedDataRef
SecCmsDigestedDataCreate(SecCmsMessageRef cmsg, SECAlgorithmID *digestalg)
    API_AVAILABLE(macos(10.4), ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#pragma clang diagnostic pop

/*!
    @function
    @abstract Destroy a digestedData object.
 */
extern void
SecCmsDigestedDataDestroy(SecCmsDigestedDataRef digd);

/*!
    @function
    @abstract Return pointer to digestedData object's contentInfo.
 */
extern SecCmsContentInfoRef
SecCmsDigestedDataGetContentInfo(SecCmsDigestedDataRef digd);

__END_DECLS

#endif /* _SECURITY_SECCMSDIGESTEDDATA_H_ */
