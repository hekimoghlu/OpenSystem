/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
    @header SecCmsDigestContext.h

    @availability 10.4 and later
    @abstract Interfaces of the CMS implementation.
    @discussion The functions here implement functions calculating digests.
 */

#ifndef _SECURITY_SECCMSDIGESTCONTEXT_H_
#define _SECURITY_SECCMSDIGESTCONTEXT_H_  1

#include <Security/SecCmsBase.h>

__BEGIN_DECLS

/*!
    @function
    @abstract Start digest calculation using all the digest algorithms in "digestalgs" in parallel.
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
extern SecCmsDigestContextRef
SecCmsDigestContextStartMultiple(SECAlgorithmID **digestalgs)
    API_AVAILABLE(macos(10.4), ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#pragma clang diagnostic pop

/*!
    @function
    @abstract Feed more data into the digest machine.
 */
extern void
SecCmsDigestContextUpdate(SecCmsDigestContextRef cmsdigcx, const unsigned char *data, size_t len);

/*!
    @function
    @abstract Cancel digesting operation in progress and destroy it.
    @discussion Cancel a DigestContext created with @link SecCmsDigestContextStartMultiple SecCmsDigestContextStartMultiple function@/link.
 */
extern void
SecCmsDigestContextCancel(SecCmsDigestContextRef cmsdigcx);

#if TARGET_OS_IPHONE
/*!
    @function
    @abstract Destroy a SecCmsDigestContextRef.
    @discussion Cancel a DigestContext created with @link SecCmsDigestContextStartMultiple SecCmsDigestContextStartMultiple function@/link after it has been used in a @link SecCmsSignedDataSetDigestContext SecCmsSignedDataSetDigestContext function@/link.
 */
extern void
SecCmsDigestContextDestroy(SecCmsDigestContextRef cmsdigcx)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#endif // TARGET_OS_IPHONE

#if TARGET_OS_OSX
/*!
 @function
 @abstract Finish the digests and put them into an array of CSSM_DATAs (allocated on arena)
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
extern OSStatus
SecCmsDigestContextFinishMultiple(SecCmsDigestContextRef cmsdigcx, SecArenaPoolRef arena,
                                  CSSM_DATA_PTR **digestsp)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);
#pragma clang diagnostic pop
#endif // TARGET_OS_OSX

__END_DECLS

#endif /* _SECURITY_SECCMSDIGESTCONTEXT_H_ */
