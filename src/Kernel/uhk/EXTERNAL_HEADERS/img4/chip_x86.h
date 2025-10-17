/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#ifndef __IMG4_CHIP_X86_H
#define __IMG4_CHIP_X86_H

#ifndef __IMG4_INDIRECT
#error "Please #include <img4/firmware.h> instead of this file directly"
#endif // __IMG4_INDIRECT

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @const IMG4_CHIP_X86
 * An Intel x86 processor which cannot be uniquely identified.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20200508
OS_EXPORT
const img4_chip_t _img4_chip_x86;
#define IMG4_CHIP_X86 (&_img4_chip_x86)
#else
#define IMG4_CHIP_X86 (img4if->i4if_v7.chip_x86)
#endif

/*!
 * @const IMG4_CHIP_X86_SOFTWARE_8012
 * A software-defined chip environment describing a virtualized x86 processor.
 * Since the virtual machine is at the mercy of the VM, support for any sort of
 * chip identity may not be available. Therefore this environment is returned
 * from {@link img4_chip_select_personalized_ap} and
 * {@link img4_chip_select_effective_ap} when it is called on a virtual machine
 * so that the appropriate chip environment is present entirely in software.
 *
 * This environment provides an equivalent software identity to that of
 * the {@link IMG4_CHIP_X86} chip environment on non-Gibraltar Macs.
 *
 * @discussion
 * Do not use this environment directly.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20200508
OS_EXPORT
const img4_chip_t _img4_chip_x86_software_8012;
#define IMG4_CHIP_X86_SOFTWARE_8012 (&_img4_chip_x86_software_8012)
#else
#define IMG4_CHIP_X86_SOFTWARE_8012 (img4if->i4if_v7.chip_x86_software_8012)
#endif

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMG4_CHIP_X86_H
