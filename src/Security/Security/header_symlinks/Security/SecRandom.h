/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
     @header SecRandom
     The functions provided in SecRandom.h implement high-level accessors
     to cryptographically secure random numbers.
*/

#ifndef _SECURITY_SECRANDOM_H_
#define _SECURITY_SECRANDOM_H_

#include <Security/SecBase.h>
#include <stdint.h>
#include <sys/types.h>

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

/*!
    @typedef SecRandomRef
    @abstract Reference to a (pseudo) random number generator.
*/
typedef const struct __SecRandom * SecRandomRef;

/* This is a synonym for NULL, if you'd rather use a named constant.   This
   refers to a cryptographically secure random number generator.  */
extern const SecRandomRef kSecRandomDefault
    __OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_2_0);

/*!
     @function SecRandomCopyBytes

     @abstract
     Return count random bytes in *bytes, allocated by the caller. It
     is critical to check the return value for error.

     @param rnd
     Only @p kSecRandomDefault is supported.

     @param count
     The number of bytes to generate.

     @param bytes
     A buffer to fill with random output.

     @result Return 0 on success, any other value on failure.

     @discussion
     If @p rnd is unrecognized or unsupported, @p kSecRandomDefault is
     used.
*/
int SecRandomCopyBytes(SecRandomRef __nullable rnd, size_t count, void *bytes)
    __attribute__ ((warn_unused_result))
    __OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_2_0);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS

#endif /* !_SECURITY_SECRANDOM_H_ */
