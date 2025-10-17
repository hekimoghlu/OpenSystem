/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
 * Secure Enclave Processor and associated handles.
 */
#ifndef __IMAGE4_API_COPROCESSOR_SEP_H
#define __IMAGE4_API_COPROCESSOR_SEP_H

#include <image4/image4.h>
#include <image4/types.h>

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @const IMAGE4_COPROCESSOR_SEP
 * The Secure Enclave Processor executing payloads signed by the Secure Boot CA.
 *
 * Handles for this environment are enumerated in the
 * {@link image4_coprocessor_handle_sep_t} type.
 */
IMAGE4_API_AVAILABLE_SPRING_2024
OS_EXPORT
const image4_coprocessor_t _image4_coprocessor_sep;
#define IMAGE4_COPROCESSOR_SEP (&_image4_coprocessor_sep)
IMAGE4_XNU_AVAILABLE_INDIRECT(_image4_coprocessor_sep);

/*!
 * @typedef image4_coprocessor_handle_sep_t
 * Handles describing supported SEP execution environments.
 *
 * @const IMAGE4_COPROCESSOR_HANDLE_SEP
 * The host's SEP environment.
 */
OS_CLOSED_ENUM(image4_coprocessor_handle_sep, image4_coprocessor_handle_t,
	IMAGE4_COPROCESSOR_HANDLE_SEP = 0,
	_IMAGE4_COPROCESSOR_HANDLE_SEP_CNT,
);

/*!
 * @const IMAGE4_COPROCESSOR_HANDLE_SEP_DEFAULT
 * The default handle for {@link IMAGE4_COPROCESSOR_SEP}. This constant enables
 * `DEFAULT` to be used as the second and third arguments to
 * {@link image4_environment_init_coproc} and
 * {@link image4_environment_new_coproc} respectively.
 */
#define IMAGE4_COPROCESSOR_HANDLE_SEP_DEFAULT \
	IMAGE4_COPROCESSOR_HANDLE_SEP

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_API_COPROCESSOR_SEP_H
