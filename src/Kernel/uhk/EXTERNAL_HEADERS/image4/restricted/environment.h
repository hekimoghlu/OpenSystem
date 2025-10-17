/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
 * Restricted environment interfaces.
 */
#ifndef __IMAGE4_API_RESTRICTED_ENVIRONMENT_H
#define __IMAGE4_API_RESTRICTED_ENVIRONMENT_H

#include <image4/image4.h>
#include <image4/environment.h>

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

#pragma mark Restricted API
/*!
 * @function image4_environment_get_firmware_chip
 * Returns the legacy chip pointer corresponding to an environment.
 *
 * @param nv
 * The environment to query.
 *
 * @result
 * Upon success, a pointer to an object which can be safely cast to an
 * `const img4_chip_t *` is returned. If the environment does not support
 * returning a legacy chip, NULL is returned.
 *
 * @availability
 * This API is restricted and should only be called via the
 * {@link image4_restricted_call} macro.
 *
 * This function first became available in restricted API version 1000; it will
 * be functionally neutered in version 2000.
 */
OS_EXPORT OS_WARN_RESULT OS_NONNULL2
const void *_Nullable
image4_environment_get_firmware_chip(
	uint32_t v,
	const image4_environment_t *nv);
#define image4_environment_get_firmware_chip(...) \
	image4_call_restricted(environment_get_firmware_chip, ## __VA_ARGS__)

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_API_RESTRICTED_ENVIRONMENT_H
