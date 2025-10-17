/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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

#include <sys/cdefs.h>
#include <stdbool.h>

#include <kern/assert.h>
#include <kern/debug.h>

#include <mach/exclaves.h>

#include <os/atomic_private.h>

#if DEVELOPMENT || DEBUG
extern unsigned int exclaves_debug;
#else
#define exclaves_debug 0
#endif /* DEVELOPMENT || DEBUG */

/* Flag values in exclaves_debug boot-arg/sysctl */
__options_closed_decl(exclaves_debug_flags, unsigned int, {
	exclaves_debug_show_errors = 0x1,
	exclaves_debug_show_progress = 0x2,
	exclaves_debug_show_scheduler_request_response = 0x4,
	exclaves_debug_show_storage_upcalls = 0x8,
	exclaves_debug_show_iokit_upcalls = 0x10,
	exclaves_debug_show_notification_upcalls = 0x20,
	exclaves_debug_show_test_output = 0x40,
	exclaves_debug_show_lifecycle_upcalls = 0x80,
});

#define EXCLAVES_ENABLE_SHOW_ERRORS                     (DEVELOPMENT || DEBUG)
#define EXCLAVES_ENABLE_SHOW_PROGRESS                   (DEVELOPMENT || DEBUG)
#define EXCLAVES_ENABLE_SHOW_SCHEDULER_REQUEST_RESPONSE (DEVELOPMENT || DEBUG)
#define EXCLAVES_ENABLE_SHOW_STORAGE_UPCALLS            (DEVELOPMENT || DEBUG)
#define EXCLAVES_ENABLE_SHOW_IOKIT_UPCALLS              (DEVELOPMENT || DEBUG)
#define EXCLAVES_ENABLE_SHOW_NOTIFICATION_UPCALLS       (DEVELOPMENT || DEBUG)
#define EXCLAVES_ENABLE_SHOW_TEST_OUTPUT                (DEVELOPMENT || DEBUG)
#define EXCLAVES_ENABLE_SHOW_LIFECYCLE_UPCALLS          (DEVELOPMENT || DEBUG)

#if EXCLAVES_ENABLE_SHOW_ERRORS || EXCLAVES_ENABLE_SHOW_TEST_OUTPUT
#define exclaves_debug_show_errors_flag (exclaves_debug_show_errors|exclaves_debug_show_test_output)
#else
#define exclaves_debug_show_errors_flag 0
#endif
#if EXCLAVES_ENABLE_SHOW_PROGRESS
#define exclaves_debug_show_progress_flag exclaves_debug_show_progress
#else
#define exclaves_debug_show_progress_flag 0
#endif
#if EXCLAVES_ENABLE_SHOW_SCHEDULER_REQUEST_RESPONSE
#define exclaves_debug_show_scheduler_request_response_flag \
    exclaves_debug_show_scheduler_request_response
#else
#define exclaves_debug_show_scheduler_request_response_flag 0
#endif
#if EXCLAVES_ENABLE_SHOW_STORAGE_UPCALLS
#define exclaves_debug_show_storage_upcalls_flag \
    exclaves_debug_show_storage_upcalls
#else
#define exclaves_debug_show_storage_upcalls_flag 0
#endif
#if EXCLAVES_ENABLE_SHOW_IOKIT_UPCALLS
#define exclaves_debug_show_iokit_upcalls_flag exclaves_debug_show_iokit_upcalls
#else
#define exclaves_debug_show_iokit_upcalls_flag 0
#endif
#if EXCLAVES_ENABLE_SHOW_NOTIFICATION_UPCALLS
#define exclaves_debug_show_notification_upcalls_flag exclaves_debug_show_notification_upcalls
#else
#define exclaves_debug_show_notification_upcalls_flag 0
#endif
#if EXCLAVES_ENABLE_SHOW_TEST_OUTPUT
#define exclaves_debug_show_test_output_flag exclaves_debug_show_test_output
#else
#define exclaves_debug_show_test_output_flag 0
#endif
#if EXCLAVES_ENABLE_SHOW_LIFECYCLE_UPCALLS
#define exclaves_debug_show_lifecycle_upcalls_flag exclaves_debug_show_lifecycle_upcalls
#else
#define exclaves_debug_show_lifecycle_upcalls_flag 0
#endif

#define exclaves_debug_enabled(flag) \
    ((bool)(exclaves_debug & exclaves_debug_##flag##_flag))
#define exclaves_debug_printf(flag, format, ...) ({ \
	if (exclaves_debug_enabled(flag)) { \
	        printf(format, ##__VA_ARGS__); \
	}})


#pragma mark exclaves relaxed requirement management

#if DEVELOPMENT || DEVELOPMENT
extern exclaves_requirement_t exclaves_relaxed_requirements;
#else
extern const exclaves_requirement_t exclaves_relaxed_requirements;
#endif /* DEVELOPMENT || DEBUG */

/*
 * Return true if the specified exclaves requirement has been relaxed, false
 * otherwise.
 */
static inline bool
exclaves_requirement_is_relaxed(exclaves_requirement_t requirement)
{
	assert3u(requirement & (requirement - 1), ==, 0);
	return (requirement & exclaves_relaxed_requirements) != 0;
}

#if DEVELOPMENT || DEBUG
static inline void
exclaves_requirement_relax(exclaves_requirement_t requirement)
{
	assert3u(requirement & (requirement - 1), ==, 0);
	os_atomic_or(&exclaves_relaxed_requirements, requirement, relaxed);
}
#else
#define exclaves_requirement_relax(req)
#endif /* DEVELOPMENT || DEBUG */
/*
 * Called when a requirement has not been met. Produces a log message and
 * continues if the requirement is relaxed, otherwise panics.
 */
#define exclaves_requirement_assert(requirement, fmt, ...) { \
	if (exclaves_requirement_is_relaxed(requirement)) {                   \
	        exclaves_debug_printf(show_errors,                            \
	            "exclaves: requirement was relaxed, ignoring error: "     \
	             fmt "\n", ##__VA_ARGS__);                                \
	} else {                                                              \
	        panic("exclaves: requirement failed: " fmt,                   \
	            ##__VA_ARGS__);                                           \
	}                                                                     \
};

#endif /* CONFIG_EXCLAVES */
