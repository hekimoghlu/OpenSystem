/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#ifndef _BTI_TELEMETRY_H_
#define _BTI_TELEMETRY_H_
#ifdef CONFIG_BTI_TELEMETRY
#include <mach/exception.h>
#include <mach/vm_types.h>
#include <mach/machine/thread_status.h>

/**
 * Wakes up the BTI exception telemetry subsystem. Call once per boot.
 */
void
bti_telemetry_init(void);

/**
 *  Handle a BTI exception. Returns TRUE if handled and OK to return from the
 *  exception, false otherwise.
 */
bool
bti_telemetry_handle_exception(arm_saved_state_t *state);

#endif /* CONFIG_BTI_TELEMETRY */
#endif /* _BTI_TELEMETRY_H_ */
