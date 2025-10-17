/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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
#include <utilities/SecCFCCWrappers.h>

#include <utilities/simulatecrash_assert.h>

CFDataRef CFDataCreateDigestWithBytes(CFAllocatorRef allocator, const struct ccdigest_info *di, size_t len,
                                const void *data, CFErrorRef *error) {
    CFMutableDataRef digest = CFDataCreateMutable(allocator, di->output_size);
    CFDataSetLength(digest, di->output_size);
    ccdigest(di, len, data, CFDataGetMutableBytePtr(digest));
    return digest;
}

CFDataRef CFDataCreateSHA1DigestWithBytes(CFAllocatorRef allocator, size_t len, const void *data, CFErrorRef *error) {
    return CFDataCreateDigestWithBytes(allocator, ccsha1_di(), len, data, error);
}

CFDataRef CFDataCreateSHA256DigestWithBytes(CFAllocatorRef allocator, size_t len, const void *data, CFErrorRef *error) {
    return CFDataCreateDigestWithBytes(allocator, ccsha256_di(), len, data, error);
}


CFDataRef CFDataCopySHA1Digest(CFDataRef dataToDigest, CFErrorRef *error) {
    CFIndex length = CFDataGetLength(dataToDigest);
    assert((unsigned long)length < UINT32_MAX); /* Debug check. Correct as long as CFIndex is long */
    return CFDataCreateSHA1DigestWithBytes(CFGetAllocator(dataToDigest), length, CFDataGetBytePtr(dataToDigest), error);
}

CFDataRef CFDataCopySHA256Digest(CFDataRef dataToDigest, CFErrorRef *error) {
    CFIndex length = CFDataGetLength(dataToDigest);
    assert((unsigned long)length < UINT32_MAX); /* Debug check. Correct as long as CFIndex is long */
    return CFDataCreateSHA256DigestWithBytes(CFGetAllocator(dataToDigest), length, CFDataGetBytePtr(dataToDigest), error);
}
