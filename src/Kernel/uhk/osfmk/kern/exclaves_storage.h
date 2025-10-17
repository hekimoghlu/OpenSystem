/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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

#if CONFIG_EXCLAVES

#if __has_include(<Tightbeam/tightbeam.h>)

#include <stdint.h>

#include <Tightbeam/tightbeam.h>
#include <Tightbeam/tightbeam_private.h>

#include "kern/exclaves.tightbeam.h"

#include "exclaves_resource.h"

__BEGIN_DECLS

/* Legacy upcall handlers */

extern tb_error_t
    exclaves_storage_upcall_legacy_root(const uint8_t exclaveid[_Nonnull 32],
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_root__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_open(const enum xnuupcalls_fstag_s fstag,
    const uint64_t rootid, const uint8_t name[_Nonnull 256],
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_open__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_close(const enum xnuupcalls_fstag_s fstag,
    const uint64_t fileid,
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_close__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_create(const enum xnuupcalls_fstag_s fstag,
    const uint64_t rootid, const uint8_t name[_Nonnull 256],
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_create__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_read(const enum xnuupcalls_fstag_s fstag,
    const uint64_t fileid, const struct xnuupcalls_iodesc_s * _Nonnull descriptor,
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_read__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_write(const enum xnuupcalls_fstag_s fstag,
    const uint64_t fileid, const struct xnuupcalls_iodesc_s * _Nonnull descriptor,
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_write__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_remove(const enum xnuupcalls_fstag_s fstag,
    const uint64_t rootid, const uint8_t name[_Nonnull 256],
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_remove__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_sync(const enum xnuupcalls_fstag_s fstag,
    const enum xnuupcalls_syncop_s op,
    const uint64_t fileid,
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_sync__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_readdir(const enum xnuupcalls_fstag_s fstag,
    const uint64_t fileid, const uint64_t buf,
    const uint32_t length,
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_readdir__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_getsize(const enum xnuupcalls_fstag_s fstag,
    const uint64_t fileid,
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_getsize__result_s));

extern tb_error_t
    exclaves_storage_upcall_legacy_sealstate(const enum xnuupcalls_fstag_s fstag,
    tb_error_t (^_Nonnull completion)(xnuupcalls_xnuupcalls_sealstate__result_s));

/* Upcall handlers */

extern tb_error_t
    exclaves_storage_upcall_root(const uint8_t exclaveid[_Nonnull 32],
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_root__result_s));

extern tb_error_t
    exclaves_storage_upcall_rootex(const uint32_t fstag,
    const uint8_t exclaveid[_Nonnull 32],
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_rootex__result_s));

extern tb_error_t
    exclaves_storage_upcall_open(const uint32_t fstag,
    const uint64_t rootid, const uint8_t name[_Nonnull 256],
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_open__result_s));

extern tb_error_t
    exclaves_storage_upcall_close(const uint32_t fstag,
    const uint64_t fileid,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_close__result_s));

extern tb_error_t
    exclaves_storage_upcall_create(const uint32_t fstag,
    const uint64_t rootid, const uint8_t name[_Nonnull 256],
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_create__result_s));

extern tb_error_t
    exclaves_storage_upcall_read(const uint32_t fstag,
    const uint64_t fileid, const struct xnuupcallsv2_iodesc_s * _Nonnull descriptor,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_read__result_s));

extern tb_error_t
    exclaves_storage_upcall_write(const uint32_t fstag,
    const uint64_t fileid, const struct xnuupcallsv2_iodesc_s * _Nonnull descriptor,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_write__result_s));

extern tb_error_t
    exclaves_storage_upcall_remove(const uint32_t fstag,
    const uint64_t rootid, const uint8_t name[_Nonnull 256],
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_remove__result_s));

extern tb_error_t
    exclaves_storage_upcall_sync(const uint32_t fstag,
    const xnuupcallsv2_syncop_s * _Nonnull op,
    const uint64_t fileid,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_sync__result_s));

extern tb_error_t
    exclaves_storage_upcall_readdir(const uint32_t fstag,
    const uint64_t fileid, const uint64_t buf,
    const uint32_t length,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_readdir__result_s));

extern tb_error_t
    exclaves_storage_upcall_getsize(const uint32_t fstag,
    const uint64_t fileid,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_getsize__result_s));

extern tb_error_t
    exclaves_storage_upcall_sealstate(const uint32_t fstag,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_sealstate__result_s));

extern tb_error_t
    exclaves_storage_upcall_queryvolumegroup(const uint8_t vguuid[_Nonnull 37],
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_storageupcallsprivate_queryvolumegroup__result_s));

__END_DECLS

#endif /* __has_include(<Tightbeam/tightbeam.h>) */

#endif /* CONFIG_EXCLAVES */
