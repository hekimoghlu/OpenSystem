/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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

#ifndef XNU_DARWINTEST_UTILS_H
#define XNU_DARWINTEST_UTILS_H

#include <stdbool.h>

/* Misc. utility functions for writing darwintests. */
bool is_development_kernel(void);

/* Launches the given helper variant as a managed process. */
pid_t launch_background_helper(
	const char* variant,
	bool start_suspended,
	bool memorystatus_managed);
/*
 * Set the process's managed bit, so that the memorystatus subsystem treats
 * this process like an app instead of a sysproc.
 */
void set_process_memorystatus_managed(pid_t pid);

#define XNU_T_META_SOC_SPECIFIC T_META_TAG("SoCSpecific")

#define XNU_T_META_REQUIRES_DEVELOPMENT_KERNEL T_META_REQUIRES_SYSCTL_EQ("kern.development", 1)
#define XNU_T_META_REQUIRES_RELEASE_KERNEL T_META_REQUIRES_SYSCTL_EQ("kern.development", 0)

#endif /* XNU_DARWINTEST_UTILS_H */
