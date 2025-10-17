/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#ifndef __IMG4_CHIP_SEP_H
#define __IMG4_CHIP_SEP_H

#ifndef __IMG4_INDIRECT
#error "Please #include <img4/firmware.h> instead of this file directly"
#endif // __IMG4_INDIRECT

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @const IMG4_CHIP_SEP_SHA1
 * The Secure Enclave Processor on an Apple ARM SoC with an embedded sha1
 * certifcate chain.
 *
 * This chip environment represents one unique instance of such a chip.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20211119
OS_EXPORT
const img4_chip_t _img4_chip_sep_sha1;
#define IMG4_CHIP_SEP_SHA1 (&_img4_chip_sep_sha1)
#else
#define IMG4_CHIP_SEP_SHA1 (img4if->i4if_v16.chip_sep_sha1)
#endif

/*!
 * @const IMG4_CHIP_SEP_SHA2_384
 * The Secure Enclave Processor on an Apple ARM SoC with an embedded sha2-384
 * certifcate chain.
 *
 * This chip environment represents one unique instance of such a chip.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20211119
OS_EXPORT
const img4_chip_t _img4_chip_sep_sha2_384;
#define IMG4_CHIP_SEP_SHA2_384 (&_img4_chip_sep_sha2_384)
#else
#define IMG4_CHIP_SEP_SHA2_384 (img4if->i4if_v16.chip_sep_sha2_384)
#endif

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMG4_CHIP_SEP_H
