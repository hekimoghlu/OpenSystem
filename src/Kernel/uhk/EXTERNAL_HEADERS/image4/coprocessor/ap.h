/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
 * Application Processor and associated handles.
 */
#ifndef __IMAGE4_API_COPROCESSOR_AP_H
#define __IMAGE4_API_COPROCESSOR_AP_H

#include <image4/image4.h>
#include <image4/types.h>

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @const IMAGE4_COPROCESSOR_AP
 * The Application Processor executing payloads signed by the Secure Boot CA.
 *
 * Handles for this environment are enumerated in the
 * {@link image4_coprocessor_ap_handle_t} type.
 */
IMAGE4_API_AVAILABLE_SPRING_2024
OS_EXPORT
const image4_coprocessor_t _image4_coprocessor_ap;
#define IMAGE4_COPROCESSOR_AP (&_image4_coprocessor_ap)
IMAGE4_XNU_AVAILABLE_INDIRECT(_image4_coprocessor_ap);

/*!
 * @typedef image4_coprocessor_handle_ap_t
 * Handles describing supported AP execution environments.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_AP
 * The host's Application Processor environment.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_AP_FF00
 * The software AP environment used for loading globally-signed OTA update brain
 * trust caches.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_AP_FF01
 * The software AP environment used for loading globally-signed Install
 * Assistant brain trust caches.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_AP_FF06
 * The software AP environment used for loading globally-signed Bootability
 * brain trust caches.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_AP_PDI
 * The sideloading AP environment used to load a personalized disk image.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_AP_SRDP
 * The sideloading AP environment used to load firmware which has been
 * authorized as part of the Security Research Device Program.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_AP_DDI
 * The sideloading AP environment used to load a personalized disk image which
 * is automatically mounted at boot.
 *
 * This handle is available starting in API version 20231027.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_AP_BOOTPC
 * The AP environment for use in calculating boot policy closures. While relying
 * parties can simply do an unauthenticated calculation to verify that a
 * manifest is consistent with an authorized closure measurement, the initial
 * generation of that authorized measurement must still verify that the boot
 * ticket was issued by the Secure Boot CA. This environment facilitates that
 * procedure.
 *
 * This handle is available starting in API version 20240318.
 */
OS_CLOSED_ENUM(image4_coprocessor_handle_ap, image4_coprocessor_handle_t,
	IMAGE4_COPROCESSOR_HANDLE_AP = 0,
	IMAGE4_COPROCESSOR_HANDLE_AP_FF00,
	IMAGE4_COPROCESSOR_HANDLE_AP_FF01,
	IMAGE4_COPROCESSOR_HANDLE_AP_FF06,
	IMAGE4_COPROCESSOR_HANDLE_AP_PDI,
	IMAGE4_COPROCESSOR_HANDLE_AP_SRDP,
	IMAGE4_COPROCESSOR_HANDLE_AP_RESERVED_0,
	IMAGE4_COPROCESSOR_HANDLE_AP_RESERVED_1,
	IMAGE4_COPROCESSOR_HANDLE_AP_RESERVED_2,
	IMAGE4_COPROCESSOR_HANDLE_AP_DDI,
	IMAGE4_COPROCESSOR_HANDLE_AP_BOOTPC,
	_IMAGE4_COPROCESSOR_HANDLE_AP_CNT,
);

/*!
 * @const IMAGE4_COPROCESSOR_HANDLE_AP_DEFAULT
 * The default handle for {@link IMAGE4_COPROCESSOR_AP}. This constant enables
 * `DEFAULT` to be used as the second and third arguments to
 * {@link image4_environment_init_coproc} and
 * {@link image4_environment_new_coproc} respectively.
 */
#define IMAGE4_COPROCESSOR_HANDLE_AP_DEFAULT IMAGE4_COPROCESSOR_HANDLE_AP

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_API_COPROCESSOR_AP_H
