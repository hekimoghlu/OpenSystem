/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#ifndef __IMG4_CHIP_AP_CATEGORY_H
#define __IMG4_CHIP_AP_CATEGORY_H

#ifndef __IMG4_INDIRECT
#error "Please #include <img4/firmware.h> instead of this file directly"
#endif // __IMG4_INDIRECT

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @const IMG4_CHIP_AP_CATEGORY_FF02
 * The Application Processor on an Intel Mac product.
 *
 * This chip environment represents one unique instance of such a chip, though
 * the uniqueness is not enforced by a secure boot chain with anti-replay
 * properties, and therefore this chip environment should be considered as
 * equivalent to a global signing environment.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20210305
OS_EXPORT
const img4_chip_t _img4_chip_ap_category_ff02;
#define IMG4_CHIP_AP_CATEGORY_FF02 (&_img4_chip_ap_category_ff02)
#else
#define IMG4_CHIP_AP_CATEGORY_FF02 (img4if->i4if_v12.chip_ap_category_ff02)
#endif

/*!
 * @const IMG4_CHIP_AP_CATEGORY_FF03
 * An Intel x86 processor whose chain of trust is rooted in an instance of a
 * {@link IMG4_CHIP_AP_SHA2_384} chip.
 *
 * This chip environment represents one unique instance of such a chip pair.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20210305
OS_EXPORT
const img4_chip_t _img4_chip_ap_category_ff03;
#define IMG4_CHIP_AP_CATEGORY_FF03 (&_img4_chip_ap_category_ff03)
#else
#define IMG4_CHIP_AP_CATEGORY_FF03 (img4if->i4if_v12.chip_ap_category_ff03)
#endif

/*!
 * @const IMG4_CHIP_AP_CATEGORY_FF04_F0
 * The Application Processor of an Apple ARM SoC in an Apple Silicon Mac
 * product.
 *
 * This chip environment represents one unique instance of such a chip.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20210305
OS_EXPORT
const img4_chip_t _img4_chip_ap_category_ff04_f0;
#define IMG4_CHIP_AP_CATEGORY_FF04_F0 (&_img4_chip_ap_category_ff04_f0)
#else
#define IMG4_CHIP_AP_CATEGORY_FF04_F0 \
		(img4if->i4if_v12.chip_ap_category_ff04_f0)
#endif

/*!
 * @const IMG4_CHIP_AP_CATEGORY_FF04_F1
 * The Application Processor of an Apple ARM SoC in an iPhone, iPad, or iPod
 * touch product.
 *
 * This chip environment represents one unique instance of such a chip.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20210305
OS_EXPORT
const img4_chip_t _img4_chip_ap_category_ff04_f1;
#define IMG4_CHIP_AP_CATEGORY_FF04_F1 (&_img4_chip_ap_category_ff04_f1)
#else
#define IMG4_CHIP_AP_CATEGORY_FF04_F1 \
		(img4if->i4if_v12.chip_ap_category_ff04_f1)
#endif

/*!
 * @const IMG4_CHIP_AP_CATEGORY_FF04_F2
 * The Application Processor of an Apple ARM SoC in an ï£¿watch product.
 *
 * This chip environment represents one unique instance of such a chip.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20210305
OS_EXPORT
const img4_chip_t _img4_chip_ap_category_ff04_f2;
#define IMG4_CHIP_AP_CATEGORY_FF04_F2 (&_img4_chip_ap_category_ff04_f2)
#else
#define IMG4_CHIP_AP_CATEGORY_FF04_F2 \
		(img4if->i4if_v12.chip_ap_category_ff04_f2)
#endif

/*!
 * @const IMG4_CHIP_AP_CATEGORY_FF04_F3
 * The Application Processor of an Apple ARM SoC in an ï£¿tv or HomePod product.
 *
 * This chip environment represents one unique instance of such a chip.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20210305
OS_EXPORT
const img4_chip_t _img4_chip_ap_category_ff04_f3;
#define IMG4_CHIP_AP_CATEGORY_FF04_F3 (&_img4_chip_ap_category_ff04_f3)
#else
#define IMG4_CHIP_AP_CATEGORY_FF04_F3 \
		(img4if->i4if_v12.chip_ap_category_ff04_f3)
#endif

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMG4_CHIP_AP_CATEGORY_H
