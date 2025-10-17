/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
 * @header
 * Second-generation virtualized Application Processor and associated handles.
 */
#ifndef __IMAGE4_API_COPROCESSOR_VMA2_H
#define __IMAGE4_API_COPROCESSOR_VMA2_H

#include <image4/image4.h>
#include <image4/types.h>

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @const IMAGE4_COPROCESSOR_VMA2
 * The second-generation virtualized Application Processor executing payloads
 * signed by the Secure Boot Extra Content CA.
 *
 * Handles for this environment are enumerated in the
 * {@link image4_coprocessor_handle_vma2_t} type.
 *
 * @discussion
 * Unlike {@link IMAGE4_COPROCESSOR_AP}, the default handle for this coprocessor
 * will not consult the host's secure boot level since virtualized APs do not
 * have a specific reduced or permissive security policy. They simply use the
 * same policy as physical SoCs.
 *
 * @availability
 * This coprocessor is available starting in API version 20240318.
 */
IMAGE4_API_AVAILABLE_SPRING_2024
OS_EXPORT
const image4_coprocessor_t _image4_coprocessor_vma2;
#define IMAGE4_COPROCESSOR_VMA2 (&_image4_coprocessor_vma2)
IMAGE4_XNU_AVAILABLE_INDIRECT(_image4_coprocessor_vma2);

/*!
 * @typedef image4_coprocessor_handle_x86_t
 * Handles describing supported second-generation virtualized AP execution
 * environments.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_VMA2
 * The personalized VMA2 environment.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_VMA2_PDI
 * The sideloading environment used to load a personalized disk image.
 *
 * This handle is available starting in API version 20240406.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_VMA2_DDI
 * The sideloading environment used to load a personalized disk image which
 * is automatically mounted at boot.
 *
 * This handle is available starting in API version 20240406.
 */
OS_CLOSED_ENUM(image4_coprocessor_handle_vma2, image4_coprocessor_handle_t,
	IMAGE4_COPROCESSOR_HANDLE_VMA2 = 0,
	IMAGE4_COPROCESSOR_HANDLE_VMA2_PDI,
	IMAGE4_COPROCESSOR_HANDLE_VMA2_DDI,
	_IMAGE4_COPROCESSOR_HANDLE_VMA2_CNT,
);

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_API_COPROCESSOR_VMA2_H
