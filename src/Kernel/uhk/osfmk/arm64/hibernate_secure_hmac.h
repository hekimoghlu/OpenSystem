/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include <sys/cdefs.h>
#include <stdint.h>
#include <stdbool.h>
#include <libkern/crypto/sha2.h>
#include <mach/vm_types.h>
#include <pexpert/arm64/board_config.h>
#if CONFIG_SPTM
#include <arm64/sptm/sptm.h>
#else
#include <arm64/ppl/ppl_hib.h>
#endif /* CONFIG_SPTM */
#include <IOKit/IOHibernatePrivate.h>

__BEGIN_DECLS

/**
 * State representing where in the hibernation process a specific secure HMAC
 * call is taking place.
 */
typedef enum {
	SECURE_HMAC_HIB_NOT_STARTED  = 0x1,
	SECURE_HMAC_HIB_SETUP        = 0x2,
	SECURE_HMAC_HIB_WRITE_IMAGE  = 0x4,
	SECURE_HMAC_HIB_RESTORE      = 0x8
} secure_hmac_hib_state_t;

void secure_hmac_init(void);
vm_address_t secure_hmac_get_reg_base(void);
vm_address_t secure_hmac_get_aes_reg_base(void);
vm_address_t secure_hmac_get_aes_offset(void);

void secure_hmac_hibernate_begin(
	secure_hmac_hib_state_t state,
	uint64_t *io_buffer_pages,
	uint32_t num_io_buffer_pages);
void secure_hmac_hibernate_end(void);

void secure_hmac_reset(secure_hmac_hib_state_t state, bool wired_pages);
int secure_hmac_update_and_compress_page(
	secure_hmac_hib_state_t state,
	ppnum_t page_number,
	const void **uncompressed,
	const void **encrypted,
	void *compressed);
void secure_hmac_final(secure_hmac_hib_state_t state, uint8_t *output, size_t output_len);
uint64_t secure_hmac_fetch_hibseg_and_info(
	/* out */ void *buffer,
	/* in */ uint64_t buffer_len,
	/* out */ IOHibernateHibSegInfo *info);
void secure_hmac_compute_rorgn_hmac(void);
void secure_hmac_fetch_rorgn_sha(uint8_t *output, size_t output_len);
void secure_hmac_fetch_rorgn_hmac(uint8_t *output, size_t output_len);
void secure_hmac_finalize_image(
	const void *image_hash,
	size_t image_hash_len,
	uint8_t *hmac,
	size_t hmac_len);
void secure_hmac_get_io_ranges(const hib_phys_range_t **io_ranges, size_t *num_io_ranges);
#if CONFIG_SPTM
bool hmac_is_io_buffer_page(uint64_t paddr);
#endif

__END_DECLS
