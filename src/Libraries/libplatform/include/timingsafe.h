/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#ifndef timingsafe_h
#define timingsafe_h
#include <Availability.h>
#include <os/base.h>
#include <stdint.h>

/**
 Token used to track state from enable() to disable().
 */
typedef uint64_t timingsafe_token_t;

__BEGIN_DECLS

/**
 @function timingsafe_enable_if_supported
 @abstract Unconditionally enable all supported timingsafe features.
 If timingsafe features aren't supported, they are ignored. If no features are
 supported, this is a no-op.

 @return The opaque token to use in timingsafe_restore_if_supported().
 */
__API_AVAILABLE(macos(15.2), ios(18.2), tvos(18.2), watchos(11.2), visionos(2.2))
OS_EXPORT OS_WARN_RESULT
OS_SWIFT_UNAVAILABLE_FROM_ASYNC("Not supported for async.")
timingsafe_token_t timingsafe_enable_if_supported(void);

/**
 @function timingsafe_restore_if_supported
 @abstract Restore timingsafe features to the state they were in before calling
 timingsafe_enable_if_supported and given the provided token.

 @param token
 The token returned by timingsafe_enable_if_supported.
 */
__API_AVAILABLE(macos(15.2), ios(18.2), tvos(18.2), watchos(11.2), visionos(2.2))
OS_EXPORT
OS_SWIFT_UNAVAILABLE_FROM_ASYNC("Not supported for async.")
void timingsafe_restore_if_supported(timingsafe_token_t token);

__END_DECLS

#endif /* timingsafe_h */
