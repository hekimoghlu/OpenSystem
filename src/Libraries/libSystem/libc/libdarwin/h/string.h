/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
 * Non-standard, Darwin-specific additions to the string(3) family of APIs.
 */
#ifndef __DARWIN_STRING_H
#define __DARWIN_STRING_H

#include <os/base.h>
#include <os/api.h>
#include <sys/cdefs.h>

#if DARWIN_TAPI
#include "tapi.h"
#endif

__BEGIN_DECLS;

/*!
 * @typedef os_flag_t
 * A type describing a flag's human-readable name.
 *
 * @property ohf_flag
 * The flag value.
 *
 * @property ohf_human
 * The human-readable, string representation of the flag value.
 */
DARWIN_API_AVAILABLE_20170407
typedef struct _os_flag {
	const uint64_t ohf_flag;
	const char *const ohf_human_flag;
} os_flag_t;

/*!
 * @define OS_FLAGSET_COUNT
 * The maximum number of flags that a flagset can represent.
 */
#define OS_FLAGSET_COUNT (sizeof(uint64_t) * BYTE_SIZE)

/*!
 * @typedef os_flagset_t
 * A type describing an array of human flags. Can accommodate up to 64 flags.
 */
DARWIN_API_AVAILABLE_20170407
typedef os_flag_t os_flagset_t[OS_FLAGSET_COUNT];

/*!
 * @macro os_flag_init
 * Initializer for a {@link os_flag_t} structure which stringifies the
 * flag value.
 *
 * @param __flag
 * The name of the flag to initialize.
 */
#define os_flag_init(__flag) { \
	.ohf_flag = __flag, \
	.ohf_human_flag = #__flag, \
}

/*!
 * @function strerror_np
 * Returns a human-readable string for the given {@link errno_t} or
 * POSIX error code.
 *
 * @param code
 * The error code for which to obtain the string.
 *
 * @result
 * A human-readable string describing the error condition. If a POSIX error code
 * is given, this is equivalent to a call to strerror(3).
 */
DARWIN_API_AVAILABLE_20170407
OS_EXPORT OS_COLD OS_WARN_RESULT OS_PURE
const char *
strerror_np(int code);

/*!
 * @function strexit_np
 * Returns a human-readable string for the given sysexits(3) code.
 *
 * @param code
 * The exit code for which to obtain the string.
 *
 * @result
 * A human-readable string describing the exit condition.
 */
DARWIN_API_AVAILABLE_20190830
OS_EXPORT OS_COLD OS_WARN_RESULT OS_PURE
const char *
strexit_np(int code);

/*!
 * @function symerror_np
 * Returns the token name of the given {@link errno_t} or POSIX error
 * code.
 *
 * @param code
 * The error code for which to obtain the token string.
 *
 * @result
 * The string describing the error token. For example, if code 2 is passed, the
 * string "EPERM" is returned.
 */
DARWIN_API_AVAILABLE_20170407
OS_EXPORT OS_COLD OS_WARN_RESULT OS_PURE
const char *
symerror_np(int code);

/*!
 * @function symexit_np
 * Returns the token name of the given sysexits(3) code.
 *
 * @param code
 * The sysexits(3) code for which to obtain the token string.
 *
 * @result
 * The string describing the exit code. For example, if 64 is passed, the string
 * "EX_USAGE" is returned. If the code is unrecognized, "EX_UNAVAILABLE" is
 * returned, which is more or less documented in sysexits(3) as the Â¯\_(ãƒ„)_/Â¯
 * exit code.
 */
DARWIN_API_AVAILABLE_20170407
OS_EXPORT OS_COLD OS_WARN_RESULT OS_PURE
const char *
symexit_np(int code);

/*!
 * @function os_flagset_copy_string
 * Returns a human-readable representation of the flags set for a given word.
 *
 * @param flagset
 * The human flagset which describes how to interpret the {@link flags}
 * parameter.
 *
 * @param flags
 * The flags to interpret.
 *
 * @result
 * The human-readable representation of all flags set in the {@link flags}
 * parameter, separated by the pipe character. The caller is responsible for
 * calling free(3) on this object when it is no longer needed.
 *
 * @discussion
 * This API is to be used in combination with {@link os_flag_init} to make
 * printing the contents of flags fields simple. For example, this code can
 * easily print a human-readable representation of the bits set in a Mach
 * message header:
 *
 *     static const os_flagset_t _mach_msgh_bits = {
 *          os_flag_init(MACH_MSGH_BITS_COMPLEX),
 *          os_flag_init(MACH_MSGH_BITS_RAISEIMP),
 *          os_flag_init(MACH_MSGH_BITS_IMPHOLDASRT),
 *    };
 *
 *    char *flags_string = os_flagset_copy_string(&_mach_msgh_bits,
 *            hdr->msgh_bits);
 *
 * For a message header with the MACH_MSGH_BITS_COMPLEX and
 * MACH_MSGH_BITS_RAISEIMP bits set, this will return the string
 *
 *     MACH_MSGH_BITS_COMPLEX|MACH_MSGH_BITS_RAISEIMP
 */
DARWIN_API_AVAILABLE_20170407
OS_EXPORT OS_WARN_RESULT OS_MALLOC
char *
os_flagset_copy_string(const os_flagset_t flagset, uint64_t flags);

__END_DECLS;

#endif // __DARWIN_STRING_H
