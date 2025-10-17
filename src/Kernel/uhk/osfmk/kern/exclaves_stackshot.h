/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#include <kern/kern_types.h>

#if CONFIG_EXCLAVES

extern void * exclaves_enter_start_label __asm__("EXCLAVES_ENTRY_START");
extern void * exclaves_enter_end_label   __asm__("EXCLAVES_ENTRY_END");

extern void * exclaves_upcall_start_label __asm__("EXCLAVES_UPCALL_START");
extern void * exclaves_upcall_end_label   __asm__("EXCLAVES_UPCALL_END");

extern void * exclaves_scheduler_request_start_label __asm__("EXCLAVES_SCHEDULER_REQUEST_START");
extern void * exclaves_scheduler_request_end_label   __asm__("EXCLAVES_SCHEDULER_REQUEST_END");

extern uintptr_t exclaves_enter_range_start;
extern uintptr_t exclaves_enter_range_end;
extern uintptr_t exclaves_upcall_range_start;
extern uintptr_t exclaves_upcall_range_end;

#endif /* CONFIG_EXCLAVES */
