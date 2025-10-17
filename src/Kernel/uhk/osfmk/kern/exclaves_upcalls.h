/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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

#include <mach/kern_return.h>
#include <mach/exclaves.h>

#include <stdbool.h>
#include <stdint.h>

__BEGIN_DECLS

extern kern_return_t
exclaves_call_upcall_handler(exclaves_id_t upcall_id);

extern kern_return_t
exclaves_upcall_init(void);

/* BEGIN IGNORE CODESTYLE */
extern bool
exclaves_upcall_in_range(uintptr_t, bool);
/* END IGNORE CODESTYLE */

__END_DECLS
