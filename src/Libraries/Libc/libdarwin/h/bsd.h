/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
 * Darwin-specific additions for FreeBSD APIs.
 */
#ifndef __DARWIN_BSD_H
#define __DARWIN_BSD_H

#include <os/base.h>
#include <os/api.h>
#include <sys/cdefs.h>
#include <sys/errno.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdbool.h>

#if DARWIN_TAPI
#include "tapi.h"
#endif

__BEGIN_DECLS;

/*!
 * @function sysctl_get_data_np
 * A convenience routine for getting a sysctl(3) property whose size is not
 * known at compile-time.
 *
 * @param mib
 * An array describing the property to manipulate. This is a "management
 * information base"-style descriptor, as described in sysctl(3).
 *
 * @param mib_cnt
 * The number of items in the MIB array.
 *
 * @param buff
 * On successful return, a pointer to a newly-allocated buffer. The caller is
 * responsible for free(3)ing this buffer when it is no longer needed.
 *
 * @param buff_len
 * On successful return, the length of the returned buffer.
 *
 * @result
 * See the sysctl(3) man page for possible return codes.
 */
DARWIN_API_AVAILABLE_20170407
OS_EXPORT OS_WARN_RESULT OS_NONNULL1 OS_NONNULL3 OS_NONNULL4
errno_t
sysctl_get_data_np(int *mib, size_t mib_cnt, void **buff, size_t *buff_len);

/*!
 * @function sysctlbyname_get_data_np
 * A convenience routine for getting a sysctl(3) property whose size is not
 * known at compile-time.
 *
 * @param mibdesc
 * An ASCII representation of the MIB vector describing the property to
 * manipulate. Each element of the vector described is separated by a '.'
 * character (e.g. "kern.ostype").
 *
 * @param buff
 * On successful return, a pointer to a newly-allocated buffer. The caller is
 * responsible for free(3)ing this buffer when it is no longer needed.
 *
 * @param buff_len
 * On successful return, the length of the returned buffer.
 *
 * @result
 * See the sysctl(3) man page for possible return codes.
 */
DARWIN_API_AVAILABLE_20170407
OS_EXPORT OS_WARN_RESULT OS_NONNULL1 OS_NONNULL2 OS_NONNULL3
errno_t
sysctlbyname_get_data_np(const char *mibdesc, void **buff, size_t *buff_len);

/*!
 * @function os_parse_boot_arg_int
 * A routine for extracting a boot-arg as an integer value that is semantically
 * similar to the PE_parse_boot_argn() kernel routine.
 *
 * @param which
 * The name of the boot-arg whose value is to be obtained.
 *
 * @param where
 * On successful return, the integer value of the given boot-arg. The caller
 * may pass NULL to simply check for the existence of a boot-arg. On failure,
 * this value is unmodified.
 *
 * @result
 * A Boolean indicating whether the named argument was found. If the discovered
 * argument value was not convertible to an integer according to the contract
 * in strtoll(3), the implementation will return false.
 *
 * @discussion
 * Boot-args are expressed with an '=' sign as a separator between the name and
 * value of an argument, e.g. "cs_enforcement_disable=1".
 */
DARWIN_API_AVAILABLE_20170407
OS_EXPORT OS_WARN_RESULT OS_NONNULL1
bool
os_parse_boot_arg_int(const char *which, int64_t *where);

/*!
 * @function os_parse_boot_arg_string
 * A routine for extracting a boot-arg's string value that is semantically
 * similar to the PE_parse_boot_argn() kernel routine.
 *
 * @param which
 * The name of the boot-arg whose value is to be obtained.
 *
 * @param where
 * The buffer in which to place the extracted value on successful return. The
 * caller may pass NULL to simply check for the existence of a boot-arg. On
 * failure, this value is unmodified.
 *
 * @param maxlen
 * The length of the {@link where} buffer. May be zero if the caller only wishes
 * to check for the existence of a boot-arg.
 *
 * @result
 * A Boolean indicating whether the named argument was found.
 */
DARWIN_API_AVAILABLE_20170407
OS_EXPORT OS_WARN_RESULT OS_NONNULL1
bool
os_parse_boot_arg_string(const char *which, char *where, size_t maxlen);

/*!
 * @function os_boot_arg_string_to_int
 * Canonically convert a boot-arg value, as a string, to an int64_t.
 *
 * @param value
 * A buffer containing a boot-arg value as a string.
 *
 * @param out_value
 * On successful return, the integer value of the given boot-arg. On failure,
 * this value is unmodified.
 *
 * @result
 * A Boolean indicating whether the boot-arg value could be parsed. If the value
 * was not convertible to an integer according to the contract in strtoll(3),
 * the implementation will return false.
 */
DARWIN_API_AVAILABLE_20210428
OS_EXPORT OS_WARN_RESULT OS_NONNULL1 OS_NONNULL2
bool
os_boot_arg_string_to_int(const char *value, int64_t *out_value);

/*!
 * @function os_parse_boot_arg_from_buffer_int
 * A routine for extracting a boot-arg as an integer value that is semantically
 * similar to the PE_parse_boot_argn() kernel routine.
 *
 * @param buffer
 * A buffer containing a complete boot-args string, such as one loaded from
 * NVRAM.
 *
 * @param which
 * The name of the boot-arg whose value is to be obtained.
 *
 * @param where
 * On successful return, the integer value of the given boot-arg. The caller
 * may pass NULL to simply check for the existence of a boot-arg. On failure,
 * this value is unmodified.
 *
 * @result
 * A Boolean indicating whether the named argument was found. If the discovered
 * argument value was not convertible to an integer according to the contract
 * in strtoll(3), the implementation will return false.
 *
 * @discussion
 * Boot-args are expressed with an '=' sign as a separator between the name and
 * value of an argument, e.g. "cs_enforcement_disable=1".
 */
DARWIN_API_AVAILABLE_20210428
OS_EXPORT OS_WARN_RESULT OS_NONNULL1 OS_NONNULL2
bool
os_parse_boot_arg_from_buffer_int(const char *buffer, const char *which, int64_t *where);

/*!
 * @function os_parse_boot_arg_from_buffer_string
 * A routine for extracting a boot-arg's string value that is semantically
 * similar to the PE_parse_boot_argn() kernel routine.
 *
 * @param buffer
 * A buffer containing a complete boot-args string, such as one loaded from
 * NVRAM.
 *
 * @param which
 * The name of the boot-arg whose value is to be obtained.
 *
 * @param where
 * The buffer in which to place the extracted value on successful return. The
 * caller may pass NULL to simply check for the existence of a boot-arg. On
 * failure, this value is unmodified.
 *
 * @param maxlen
 * The length of the {@link where} buffer. May be zero if the caller only wishes
 * to check for the existence of a boot-arg.
 *
 * @result
 * A Boolean indicating whether the named argument was found.
 */
DARWIN_API_AVAILABLE_20210428
OS_EXPORT OS_WARN_RESULT OS_NONNULL1 OS_NONNULL2
bool
os_parse_boot_arg_from_buffer_string(const char *buffer, const char *which, char *where, size_t maxlen);

/*!
 * @typedef os_boot_arg_enum_t
 *
 * @param context
 * Any context passed to @c os_enumerate_boot_args() or to
 * @c os_enumerate_boot_args_from_buffer().
 *
 * @param which
 * The name of the boot argument currently being enumerated.
 *
 * @param value
 * The value of the boot argument. This value may be @c NULL if the boot
 * argument does not specify a value.
 *
 * @param is_boolean
 * This boot argument is a boolean argument rather than a string argument. For
 * instance, the boot argument @c -hello_world is a boolean argument. Its mere
 * presence indicates a @c true value.
 *
 * @returns
 * @c true to continue enumeration, @c false to stop.
 *
 * @discussion
 * @a which and @a value are, when not @c NULL, pointers into data structures
 * owned by the enumeration function and must not be used outside the scope of
 * this function. Copy them with @c strdup() or equivalent if you need to
 * persist them.
 */
DARWIN_API_AVAILABLE_20210428
OS_NONNULL2
typedef bool (* os_boot_arg_enumerator_t)(void *context, const char *which, const char *value, bool is_boolean);

/*!
 * @function os_enumerate_boot_args
 * A routine for enumerating over the boot arguments present on the current
 * system.
 *
 * @param context
 * A caller-supplied context pointer that is passed to @a fp.
 *
 * @param fp
 * A function pointer to call for each enumerated boot argument.
 *
 * @discussion
 * This function can be used to walk the current system's boot-args string and
 * enumerate each of the parseable boot arguments it contains.
 */
DARWIN_API_AVAILABLE_20210428
OS_EXPORT OS_NONNULL2
void
os_enumerate_boot_args(void *context, os_boot_arg_enumerator_t fp);

/*!
 * @function os_enumerate_boot_args_from_buffer
 * A routine for enumerating over the boot arguments present in a boot-args
 * string value.
 *
 * @param buffer
 * A buffer containing a complete boot-args string, such as one loaded from
 * NVRAM.
 *
 * @param context
 * A caller-supplied context pointer that is passed to @a fp.
 *
 * @param fp
 * A function pointer to call for each enumerated boot argument.
 *
 * @discussion
 * This function can be used to walk a complete boot-args string and enumerate
 * each of the parseable boot arguments it contains.
 */
DARWIN_API_AVAILABLE_20210428
OS_EXPORT OS_NONNULL1 OS_NONNULL3
void
os_enumerate_boot_args_from_buffer(const char *buffer, void *context, os_boot_arg_enumerator_t fp);

#ifdef __BLOCKS__
/*!
 * @typedef os_boot_arg_enumerator_b_t
 *
 * @param which
 * The name of the boot argument currently being enumerated.
 *
 * @param value
 * The value of the boot argument. This value may be @c NULL if the boot
 * argument does not specify a value.
 *
 * @param is_boolean
 * This boot argument is a boolean argument rather than a string argument. For
 * instance, the boot argument @c -hello_world is a boolean argument. Its mere
 * presence indicates a @c true value.
 *
 * @returns
 * @c true to continue enumeration, @c false to stop.
 *
 * @discussion
 * @a which and @a value are, when not @c NULL, pointers into data structures
 * owned by the enumeration function and must not be used outside the scope of
 * this block. Copy them with @c strdup() or equivalent if you need to persist
 * them.
 */
DARWIN_API_AVAILABLE_20210428
OS_NONNULL1
typedef bool (^ os_boot_arg_enumerator_b_t)(const char *which, const char *value, bool is_boolean);

/*!
 * @function os_enumerate_boot_args_b
 * A routine for enumerating over the boot arguments present on the current
 * system.
 *
 * @param block
 * A block to call for each enumerated boot argument.
 *
 * @discussion
 * This function can be used to walk the current system's boot-args string and
 * enumerate each of the parseable boot arguments it contains.
 */
DARWIN_API_AVAILABLE_20210428
OS_EXPORT OS_NONNULL1
void
os_enumerate_boot_args_b(OS_NOESCAPE os_boot_arg_enumerator_b_t block);

/*!
 * @function os_enumerate_boot_args_from_buffer_b
 * A routine for enumerating over the boot arguments present in a boot-args
 * string value.
 *
 * @param buffer
 * A buffer containing a complete boot-args string, such as one loaded from
 * NVRAM.
 *
 * @param block
 * A block to call for each enumerated boot argument.
 *
 * @discussion
 * This function can be used to walk a complete boot-args string and enumerate
 * each of the parseable boot arguments it contains.
 */
DARWIN_API_AVAILABLE_20210428
OS_EXPORT OS_NONNULL1 OS_NONNULL2
void
os_enumerate_boot_args_from_buffer_b(const char *buffer, OS_NOESCAPE os_boot_arg_enumerator_b_t block);
#endif /* __BLOCKS__ */

__END_DECLS;

#endif // __DARWIN_BSD_H
