/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#pragma once

/**
 * @file system_properties.h
 * @brief System properties.
 */

#include <sys/cdefs.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

__BEGIN_DECLS

/** An opaque structure representing a system property. */
typedef struct prop_info prop_info;

/**
 * The limit on the length of a property value.
 * (See PROP_NAME_MAX for property names.)
 */
#define PROP_VALUE_MAX  92

/**
 * Sets system property `name` to `value`, creating the system property if it doesn't already exist.
 *
 * Returns 0 on success, or -1 on failure.
 */
int __system_property_set(const char* _Nonnull __name, const char* _Nonnull __value);

/**
 * Returns a `prop_info` corresponding system property `name`, or nullptr if it doesn't exist.
 * Use __system_property_read_callback() to query the current value.
 *
 * Property lookup is expensive, so it can be useful to cache the result of this
 * function rather than using __system_property_get().
 */
const prop_info* _Nullable __system_property_find(const char* _Nonnull __name);

/**
 * Calls `callback` with a consistent trio of name, value, and serial number
 * for property `pi`.
 *
 * Available since API level 26.
 */

#if __BIONIC_AVAILABILITY_GUARD(26)
void __system_property_read_callback(const prop_info* _Nonnull __pi,
    void (* _Nonnull __callback)(void* _Nullable __cookie, const char* _Nonnull __name, const char* _Nonnull __value, uint32_t __serial),
    void* _Nullable __cookie) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


/**
 * Passes a `prop_info` for each system property to the provided
 * callback. Use __system_property_read_callback() to read the value of
 * any of the properties.
 *
 * This method is for inspecting and debugging the property system, and not generally useful.
 *
 * Returns 0 on success, or -1 on failure.
 */
int __system_property_foreach(void (* _Nonnull __callback)(const prop_info* _Nonnull __pi, void* _Nullable __cookie), void* _Nullable __cookie);

/**
 * Waits for the specific system property identified by `pi` to be updated
 * past `old_serial`. Waits no longer than `relative_timeout`, or forever
 * if `relative_timeout` is null.
 *
 * If `pi` is null, waits for the global serial number instead.
 *
 * If you don't know the current serial, use 0.
 *
 * Returns true and updates `*new_serial_ptr` on success, or false if the call
 * timed out.
 *
 * Available since API level 26.
 */
struct timespec;

#if __BIONIC_AVAILABILITY_GUARD(26)
bool __system_property_wait(const prop_info* _Nullable __pi, uint32_t __old_serial, uint32_t* _Nonnull __new_serial_ptr, const struct timespec* _Nullable __relative_timeout)
    __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


/**
 * Deprecated: there's no limit on the length of a property name since
 * API level 26, though the limit on property values (PROP_VALUE_MAX) remains.
 */
#define PROP_NAME_MAX   32

/** Deprecated. Use __system_property_foreach() instead. */
const prop_info* _Nullable __system_property_find_nth(unsigned __n);
/** Deprecated. Use __system_property_read_callback() instead. */
int __system_property_read(const prop_info* _Nonnull __pi, char* _Nullable __name, char* _Nonnull __value);
/** Deprecated. Use __system_property_read_callback() instead. */
int __system_property_get(const char* _Nonnull __name, char* _Nonnull __value);
/** Deprecated: use __system_property_wait() instead. */
uint32_t __system_property_wait_any(uint32_t __old_serial);

/**
 * Reads the global serial number of the system properties _area_.
 *
 * Called to predict if a series of cached __system_property_find()
 * objects will have seen __system_property_serial() values change.
 * Also aids the converse, as changes in the global serial can
 * also be used to predict if a failed __system_property_find()
 * could in turn now find a new object; thus preventing the
 * cycles of effort to poll __system_property_find().
 *
 * Typically called at beginning of a cache cycle to signal if _any_ possible
 * changes have occurred since last. If there is, one may check each individual
 * __system_property_serial() to confirm dirty, or __system_property_find()
 * to check if the property now exists. If a call to __system_property_add()
 * or __system_property_update() has completed between two calls to
 * __system_property_area_serial() then the second call will return a larger
 * value than the first call. Beware of race conditions as changes to the
 * properties are not atomic, the main value of this call is to determine
 * whether the expensive __system_property_find() is worth retrying to see if
 * a property now exists.
 *
 * Returns the serial number on success, -1 on error.
 */
uint32_t __system_property_area_serial(void);

/**
 * Reads the serial number of a specific system property previously returned by
 * __system_property_find(). This is a cheap way to check whether a system
 * property has changed or not.
 *
 * Returns the serial number on success, -1 on error.
 */
uint32_t __system_property_serial(const prop_info* _Nonnull __pi);

//
// libc implementation detail.
//

/**
 * Initializes the system properties area in read-only mode.
 *
 * This is called automatically during libc initialization,
 * so user code should never need to call this.
 *
 * Returns 0 on success, -1 otherwise.
 */
int __system_properties_init(void);

//
// init implementation details.
//

#define PROP_SERVICE_NAME "property_service"
#define PROP_SERVICE_FOR_SYSTEM_NAME "property_service_for_system"
#define PROP_DIRNAME "/dev/__properties__"

// Messages sent to init.
#define PROP_MSG_SETPROP 1
#define PROP_MSG_SETPROP2 0x00020001

// Status codes returned by init (but not passed from libc to the caller).
#define PROP_SUCCESS 0
#define PROP_ERROR_READ_CMD 0x0004
#define PROP_ERROR_READ_DATA 0x0008
#define PROP_ERROR_READ_ONLY_PROPERTY 0x000B
#define PROP_ERROR_INVALID_NAME 0x0010
#define PROP_ERROR_INVALID_VALUE 0x0014
#define PROP_ERROR_PERMISSION_DENIED 0x0018
#define PROP_ERROR_INVALID_CMD 0x001B
#define PROP_ERROR_HANDLE_CONTROL_MESSAGE 0x0020
#define PROP_ERROR_SET_FAILED 0x0024

/**
 * Initializes the area to be used to store properties.
 *
 * Can only be done by the process that has write access to the property area,
 * typically init.
 *
 * See __system_properties_init() for the equivalent for all other processes.
 */
int __system_property_area_init(void);

/**
 * Adds a new system property.
 * Can only be done by the process that has write access to the property area --
 * typically init -- which must handle sequencing to ensure that only one property is
 * updated at a time.
 *
 * Returns 0 on success, -1 if the property area is full.
 */
int __system_property_add(const char* _Nonnull __name, unsigned int __name_length, const char* _Nonnull __value, unsigned int __value_length);

/**
 * Updates the value of a system property returned by __system_property_find().
 * Can only be done by the process that has write access to the property area --
 * typically init -- which must handle sequencing to ensure that only one property is
 * updated at a time.
 *
 * Returns 0 on success, -1 if the parameters are incorrect.
 */
int __system_property_update(prop_info* _Nonnull __pi, const char* _Nonnull __value, unsigned int __value_length);

/**
 * Reloads the system properties from disk.
 * Not intended for use by any apps except the Zygote.
 * Should only be called from the main thread.
 *
 * Pointers received from functions such as __system_property_find()
 * may be invalidated by calls to this function.
 *
 * Returns 0 on success, -1 otherwise.
 *
 * Available since API level 35.
 */

#if __BIONIC_AVAILABILITY_GUARD(35)
int __system_properties_zygote_reload(void) __INTRODUCED_IN(35);
#endif /* __BIONIC_AVAILABILITY_GUARD(35) */


/**
 * Deprecated: previously for testing, but now that SystemProperties is its own
 * testable class, there is never a reason to call this function and its
 * implementation simply returns -1.
 */
int __system_property_set_filename(const char* _Nullable __unused __filename);

__END_DECLS
