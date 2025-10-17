/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#if CONFIG_EXCLAVES

#pragma once

#include <Tightbeam/tightbeam.h>
#include <mach/kern_return.h>
#include <stdint.h>

#include "kern/exclaves.tightbeam.h"

typedef struct conclave_sharedbuffer_t {
	uint64_t physaddr[2];
} conclave_sharedbuffer_t;

__BEGIN_DECLS

extern kern_return_t
exclaves_conclave_launcher_init(uint64_t id, tb_client_connection_t *connection);

extern kern_return_t
exclaves_conclave_launcher_suspend(const tb_client_connection_t connection,
    bool suspend);

extern kern_return_t
exclaves_conclave_launcher_launch(const tb_client_connection_t connection);

extern kern_return_t
exclaves_conclave_launcher_stop(const tb_client_connection_t connection,
    uint32_t stop_reason);

/* Legacy upcall handlers */

extern tb_error_t
    exclaves_conclave_upcall_legacy_suspend(const uint32_t flags,
    tb_error_t (^completion)(xnuupcalls_xnuupcalls_conclave_suspend__result_s));

extern tb_error_t
    exclaves_conclave_upcall_legacy_stop(const uint32_t flags,
    tb_error_t (^completion)(xnuupcalls_xnuupcalls_conclave_stop__result_s));

extern tb_error_t
    exclaves_conclave_upcall_legacy_crash_info(const xnuupcalls_conclavesharedbuffer_s * shared_buf,
    const uint32_t length,
    tb_error_t (^completion)(xnuupcalls_xnuupcalls_conclave_crash_info__result_s));

/* Upcall handlers */

extern tb_error_t
    exclaves_conclave_upcall_suspend(const uint32_t flags,
    tb_error_t (^completion)(xnuupcallsv2_conclaveupcallsprivate_suspend__result_s));

extern tb_error_t
    exclaves_conclave_upcall_stop(const uint32_t flags,
    tb_error_t (^completion)(xnuupcallsv2_conclaveupcallsprivate_stop__result_s));

extern tb_error_t
    exclaves_conclave_upcall_crash_info(const xnuupcallsv2_conclavesharedbuffer_s * shared_buf,
    const uint32_t length,
    tb_error_t (^completion)(xnuupcallsv2_conclaveupcallsprivate_crashinfo__result_s));

__END_DECLS

#endif /* CONFIG_EXCLAVES */
