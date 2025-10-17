/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
#ifndef __IMG4_CHIP_AP_SOFTWARE_H
#define __IMG4_CHIP_AP_SOFTWARE_H

#ifndef __IMG4_INDIRECT
#error "Please #include <img4/firmware.h> instead of this file directly"
#endif // __IMG4_INDIRECT

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @const IMG4_CHIP_AP_SOFTWARE_FF00
 * A software-defined chip environment whose firmwares are executed on any
 * Application Processor on an Apple ARM SoC. The firmwares are loadable trust
 * caches shipped with OTA update brains.
 *
 * This chip cannot be uniquely identified.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20200508
OS_EXPORT
const img4_chip_t _img4_chip_ap_software_ff00;
#define IMG4_CHIP_AP_SOFTWARE_FF00 (&_img4_chip_ap_software_ff00)
#else
#define IMG4_CHIP_AP_SOFTWARE_FF00 (img4if->i4if_v7.chip_ap_software_ff00)
#endif

/*!
 * @const IMG4_CHIP_AP_SOFTWARE_FF01
 * A software-defined chip environment whose firmwares are executed on any
 * Application Processor on an Apple ARM SoC. The firmwares are loadable trust
 * caches which are shipped in the Install Assistant and loaded by an
 * unprivileged trampoline.
 *
 * This chip cannot be uniquely identified.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20200508
OS_EXPORT
const img4_chip_t _img4_chip_ap_software_ff01;
#define IMG4_CHIP_AP_SOFTWARE_FF01 (&_img4_chip_ap_software_ff01)
#else
#define IMG4_CHIP_AP_SOFTWARE_FF01 (img4if->i4if_v7.chip_ap_software_ff01)
#endif

/*!
 * @const IMG4_CHIP_AP_SOFTWARE_FF06
 * A software-defined chip environment whose firmwares are executed on any
 * Application Processor on an Apple ARM SoC. The firmwares are loadable trust
 * caches which are shipped in the preboot volume.
 *
 * This chip cannot be uniquely identified.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20210113
OS_EXPORT
const img4_chip_t _img4_chip_ap_software_ff06;
#define IMG4_CHIP_AP_SOFTWARE_FF06 (&_img4_chip_ap_software_ff06)
#else
#define IMG4_CHIP_AP_SOFTWARE_FF06 (img4if->i4if_v11.chip_ap_software_ff06)
#endif

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMG4_CHIP_AP_SOFTWARE_H
