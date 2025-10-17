/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
#ifndef _SECCFCCWRAPPERS_H_
#define _SECCFCCWRAPPERS_H_

#include <CoreFoundation/CoreFoundation.h>

#include <corecrypto/ccsha1.h>
#include <corecrypto/ccsha2.h>

__BEGIN_DECLS

CFDataRef CFDataCreateDigestWithBytes(CFAllocatorRef allocator, const struct ccdigest_info *di, size_t len,
                                      const void *data, CFErrorRef *error);

CFDataRef CFDataCreateSHA1DigestWithBytes(CFAllocatorRef allocator, size_t len, const void *data, CFErrorRef *error);
CFDataRef CFDataCreateSHA256DigestWithBytes(CFAllocatorRef allocator, size_t len, const void *data, CFErrorRef *error);

CFDataRef CFDataCopySHA1Digest(CFDataRef dataToDigest, CFErrorRef *error);
CFDataRef CFDataCopySHA256Digest(CFDataRef dataToDigest, CFErrorRef *error);

__END_DECLS

#endif /* _SECCFWRAPPERS_H_ */
