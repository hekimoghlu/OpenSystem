/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#ifndef _CORECRYPTO_CC_IMPL_H_
#define _CORECRYPTO_CC_IMPL_H_

#define CC_IMPL_LIST                                        \
    CC_IMPL_ITEM(UNKNOWN, 0)                                \
                                                            \
    CC_IMPL_ITEM(SHA256_LTC, 1)                             \
    CC_IMPL_ITEM(SHA256_VNG_ARM, 2)                         \
    CC_IMPL_ITEM(SHA256_VNG_ARM64_NEON, 3)                  \
    CC_IMPL_ITEM(SHA256_VNG_INTEL_SUPPLEMENTAL_SSE3, 4)     \
    CC_IMPL_ITEM(SHA256_VNG_INTEL_AVX1, 5)                  \
    CC_IMPL_ITEM(SHA256_VNG_INTEL_AVX2, 6)                  \
                                                            \
    CC_IMPL_ITEM(AES_ECB_LTC, 11)                           \
    CC_IMPL_ITEM(AES_ECB_ARM, 12)                           \
    CC_IMPL_ITEM(AES_ECB_INTEL_OPT, 13)                     \
    CC_IMPL_ITEM(AES_ECB_INTEL_AESNI, 14)                   \
    CC_IMPL_ITEM(AES_ECB_SKG, 15)                           \
    CC_IMPL_ITEM(AES_ECB_TRNG, 16)                          \
                                                            \
    CC_IMPL_ITEM(AES_XTS_GENERIC, 21)                       \
    CC_IMPL_ITEM(AES_XTS_ARM, 22)                           \
    CC_IMPL_ITEM(AES_XTS_INTEL_OPT, 23)                     \
    CC_IMPL_ITEM(AES_XTS_INTEL_AESNI, 24)                   \
                                                            \
    CC_IMPL_ITEM(SHA1_LTC, 31)                              \
    CC_IMPL_ITEM(SHA1_VNG_ARM, 32)                          \
    CC_IMPL_ITEM(SHA1_VNG_INTEL_SUPPLEMENTAL_SSE3, 33)      \
    CC_IMPL_ITEM(SHA1_VNG_INTEL_AVX1, 34)                   \
    CC_IMPL_ITEM(SHA1_VNG_INTEL_AVX2, 35)                   \
                                                            \
    CC_IMPL_ITEM(SHA384_LTC, 41)                            \
    CC_IMPL_ITEM(SHA384_VNG_ARM, 42)                        \
    CC_IMPL_ITEM(SHA384_VNG_INTEL_SUPPLEMENTAL_SSE3, 43)    \
    CC_IMPL_ITEM(SHA384_VNG_INTEL_AVX1, 44)                 \
    CC_IMPL_ITEM(SHA384_VNG_INTEL_AVX2, 45)                 \
                                                            \
    CC_IMPL_ITEM(SHA512_LTC, 51)                            \
    CC_IMPL_ITEM(SHA512_VNG_ARM, 52)                        \
    CC_IMPL_ITEM(SHA512_VNG_INTEL_SUPPLEMENTAL_SSE3, 53)    \
    CC_IMPL_ITEM(SHA512_VNG_INTEL_AVX1, 54)                 \
    CC_IMPL_ITEM(SHA512_VNG_INTEL_AVX2, 55)


#define CC_IMPL_ITEM(k, v)                      \
    CC_IMPL_##k = v,

typedef enum cc_impl {
    CC_IMPL_LIST
} cc_impl_t;

#undef CC_IMPL_ITEM

const char *cc_impl_name(cc_impl_t impl);

#endif /* _CORECRYPTO_CC_IMPL_H_ */
