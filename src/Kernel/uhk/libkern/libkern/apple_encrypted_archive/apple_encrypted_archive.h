/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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
#ifndef __APPLE_ENCRYPTED_ARCHIVE_H
#define __APPLE_ENCRYPTED_ARCHIVE_H

#include <stdint.h>
#include <os/base.h>
#include <sys/cdefs.h>
#include <sys/_types/_ssize_t.h>

/* Callbacks used to write/read data to/from the encrypted stream */
typedef ssize_t (*apple_encrypted_archive_pwrite_proc)(
	void *arg,
	const void *buf,
	size_t nbyte,
	off_t offset);

typedef ssize_t (*apple_encrypted_archive_pread_proc)(
	void *arg,
	void *buf,
	size_t nbyte,
	off_t offset);

/**
 * @abstract Get state size
 *
 * @return Positive state size (bytes) on success, and 0 on failure
 */
typedef size_t (*apple_encrypted_archive_get_state_size)(void);

/**
 * @abstract Initialize state
 *
 * @param state Encryption state buffer, \p state_size bytes
 * @param state_size Size allocated in \p state, must be at least apple_encrypted_archive_get_state_size()
 * @param recipient_public_key x9.63 encoded public key, must be on the P256 elliptic curve
 * @param recipient_public_key_size bytes stored in \p public_key (must be 65)
 *
 * @return 0 on success, and a negative error code on failure
 */
typedef int (*apple_encrypted_archive_initialize_state)(
	void *state,
	size_t state_size,
	const uint8_t *recipient_public_key,
	size_t recipient_public_key_size);

/**
 * @abstract Open encryption stream
 *
 * @discussion State must have been initialized with apple_encrypted_archive_initialize_state()
 *
 * @param state Encryption state buffer, \p state_size bytes
 * @param state_size Size allocated in \p state, must be at least apple_encrypted_archive_get_state_size()
 * @param callback_arg Value passed as first argument to the pwrite/pread callbacks
 * @param pwrite_callback Function used to write data to the encrypted stream
 * @param pread_callback Function used to read data from the encrypted stream
 *
 * @return 0 on success, and a negative error code on failure
 */
typedef int (*apple_encrypted_archive_open)(
	void *state,
	size_t state_size,
	void *callback_arg,
	apple_encrypted_archive_pwrite_proc pwrite_callback,
	apple_encrypted_archive_pread_proc pread_callback);

/**
 * @abstract Write data to encryption stream
 *
 * @discussion Stream must have been opened with apple_encrypted_archive_open()
 *
 * @param state Encryption state buffer, \p state_size bytes
 * @param state_size Size allocated in \p state, must be at least apple_encrypted_archive_get_state_size()
 * @param buf Data to write, \p nbyte bytes
 * @param nbyte Number of bytes to write from \p buf
 *
 * @return Number of bytes written on success, and a negative error code on failure
 */
typedef ssize_t (*apple_encrypted_archive_write)(
	void *state,
	size_t state_size,
	const void *buf,
	size_t nbyte);

/**
 * @abstract Close encryption stream
 *
 * @discussion Stream must have been opened with apple_encrypted_archive_open()
 *
 * @param state Encryption state buffer, \p state_size bytes
 * @param state_size Size allocated in \p state, must be at least apple_encrypted_archive_get_state_size()
 *
 * @return 0 on success, and a negative error code on failure
 */
typedef int (*apple_encrypted_archive_close)(
	void *state,
	size_t state_size);

typedef struct _apple_encrypted_archive {
	apple_encrypted_archive_get_state_size   aea_get_state_size;
	apple_encrypted_archive_initialize_state aea_initialize_state;
	apple_encrypted_archive_open             aea_open;
	apple_encrypted_archive_write            aea_write;
	apple_encrypted_archive_close            aea_close;
} apple_encrypted_archive_t;

__BEGIN_DECLS

/**
 * @abstract The AppleEncryptedArchive interface that was registered.
 */
extern const apple_encrypted_archive_t * apple_encrypted_archive;

/**
 * @abstract Registers the AppleEncryptedArchive kext interface for use within the kernel proper.
 *
 * @param aea The interface to register.
 *
 * @discussion
 * This routine may only be called once and must be called before late-const has
 * been applied to kernel memory.
 */
OS_EXPORT OS_NONNULL1
void apple_encrypted_archive_interface_register(const apple_encrypted_archive_t *aea);

#if PRIVATE

typedef void (*registration_callback_t)(void);

void apple_encrypted_archive_interface_set_registration_callback(registration_callback_t callback);

#endif /* PRIVATE */

__END_DECLS

#endif // __APPLE_ENCRYPTED_ARCHIVE_H
