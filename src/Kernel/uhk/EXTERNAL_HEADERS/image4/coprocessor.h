/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
 * Supported coprocessors.
 */
#ifndef __IMAGE4_API_COPROCESSOR_H
#define __IMAGE4_API_COPROCESSOR_H

#include <image4/image4.h>
#include <image4/types.h>

/*!
 * @section TAPI
 * TAPI doesn't like this because it sort of functions as an umbrella header
 * rather than each of these sub-headers being self-contained. But we don't want
 * to specify this as the public umbrella header because it isn't. We just do
 * this because these coprocessor definitions used to all be in this header, but
 * then it started getting crowded, so we broke them out and didn't want to
 * break dependent projects, which made TAPI upset.
 *
 * So we just don't tell it about the umbrella nature here, since these headers
 * are all self-contained; it's just that we have to make their content
 * available through just an inclusion of this header.
 */
#if !IMAGE4_INSTALLAPI
#include <image4/coprocessor/ap.h>
#include <image4/coprocessor/ap_local.h>
#include <image4/coprocessor/bootpc.h>
#include <image4/coprocessor/cryptex1.h>
#include <image4/coprocessor/sep.h>
#include <image4/coprocessor/vma2.h>
#include <image4/coprocessor/vma3.h>
#include <image4/coprocessor/x86.h>
#endif

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

#pragma mark Definitions
/*!
 * @const IMAGE4_COPROCESSOR_ARRAY_CNT
 * The maximum number of coprocessors that can be represented in an array given
 * to {@link image4_coprocessor_resolve_from_manifest}.
 */
#define IMAGE4_COPROCESSOR_ARRAY_CNT (3u)

#pragma mark Host Coprocessor
/*!
 * @const IMAGE4_COPROCESSOR_HOST
 * The host execution environment. This environment does not support handles.
 */
IMAGE4_API_AVAILABLE_SPRING_2024
OS_EXPORT
const image4_coprocessor_t _image4_coprocessor_host;
#define IMAGE4_COPROCESSOR_HOST (&_image4_coprocessor_host)
IMAGE4_XNU_AVAILABLE_INDIRECT(_image4_coprocessor_host);

/*!
 * @const IMAGE4_COPROCESSOR_HANDLE_HOST_DEFAULT
 * The default handle for {@link IMAGE4_COPROCESSOR_HOST}. This constant enables
 * `DEFAULT` to be used as the second and third arguments to
 * {@link image4_environment_init_coproc} and
 * {@link image4_environment_new_coproc} respectively.
 */
#define IMAGE4_COPROCESSOR_HANDLE_HOST_DEFAULT 0

#pragma mark API
/*!
 * @function image4_coprocessor_resolve_from_manifest
 * Resolves the coprocessor environment associated with a manifest.
 *
 * @param manifest
 * A pointer to the Image4 manifest bytes. This buffer may refer to a stitched
 * manifest and payload object, in which case the implementation will extract
 * the manifest portion.
 *
 * @param manifest_len
 * The length of the buffer referenced by {@link manifest}.
 *
 * @param coprocs
 * The list of coprocessors which could possibly authenticate the manifest. This
 * list should be kept as small as possible.
 *
 * @result
 * The coprocessor environment that can be used to authenticate the manifest, or
 * NULL if none of the provided coprocessors could be used.
 *
 * @discussion
 * Generally speaking, callers should have a priori, static knowledge of the
 * environment in which they authenticate payloads. If the caller is responsible
 * for handling payloads for multiple coprocessors, it should make the decision
 * of which coprocessor to use based on static environmental properties or
 * properties that have been forwarded from the previous stage of boot.
 *
 * This interface's existence is a conceit that this is not always possible for
 * certain trust evaluations, e.g. evaluations that need to evaluate content
 * provided by another execution context in order to counter-sign it. In such
 * cases, multiple different manifests for multiple different coprocessor
 * environments may need to be evaluated.
 *
 * This interface intentionally does not attempt to resolve a coprocessor
 * handle. The caller must still possess static knowledge of which handle must
 * be used for which coprocessor.
 *
 * @availability
 * This function first became available in API version 20240216.
 */
IMAGE4_API_AVAILABLE_FALL_2024
OS_EXPORT OS_WARN_RESULT OS_NONNULL1 OS_NONNULL3
const image4_coprocessor_t *_Nullable
image4_coprocessor_resolve_from_manifest(
	const void *__sized_by(manifest_len) manifest,
	size_t manifest_len,
	const image4_coprocessor_t *_Nullable coprocs[
		_Nonnull __static_size IMAGE4_COPROCESSOR_ARRAY_CNT]);
IMAGE4_XNU_AVAILABLE_DIRECT(image4_coprocessor_resolve_from_manifest);

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_API_COPROCESSOR_H
