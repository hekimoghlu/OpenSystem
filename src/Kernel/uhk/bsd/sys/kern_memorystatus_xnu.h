/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#include <stdbool.h>
#include <stdint.h>
#include <sys/_types/_pid_t.h>

#pragma once

#if !XNU_KERNEL_PRIVATE
#error "Including xnu-private header in unexpected target"
#endif /* !XNU_KERNEL_PRIVATE */

__BEGIN_DECLS

/* TODO: migrate other xnu-private interfaces from kern_memorystatus.h */

/*
 * Return the minimum number of available pages jetsam requires before it
 * begins killing non-idle processes. This is useful for some pageout
 * mechanisms to avoid deadlock.
 */
extern uint32_t memorystatus_get_critical_page_shortage_threshold(void);

/*
 * Return the minimum number of available pages jetsam requires before it
 * begins killing idle processes. This is consumed by the vm pressure
 * notification system in the absence of the compressor.
 */
extern uint32_t memorystatus_get_idle_exit_page_shortage_threshold(void);

/*
 * Return the minimum number of available pages jetsam requires before it
 * begins killing processes which have violated their soft memory limit. This
 * is consumed by the vm pressure notification system in the absence of the
 * compressor.
 */
extern uint32_t memorystatus_get_soft_memlimit_page_shortage_threshold(void);

/*
 * Return the minumum number of available pages jetsam requires before it
 * begins reaping long-idle processes.
 */
extern uint32_t memorystatus_get_reaper_page_shortage_threshold(void);

/*
 * Return the current number of available pages in the system.
 */
extern uint32_t memorystatus_get_available_page_count(void);

/*
 * Set the available page count and consider engaging response measures (e.g.
 * waking jetsam thread/pressure-notification thread).
 */
extern void memorystatus_update_available_page_count(uint32_t available_pages);

/*
 * Override fast-jetsam support. If override is enabled, fast-jetsam will be
 * disabled.
 */
extern void memorystatus_fast_jetsam_override(bool enable_override);

/*
 * Callout to jetsam. If pid is -1, we wake up the memorystatus thread to do asynchronous kills.
 * For any other pid we try to kill that process synchronously.
 */
extern bool memorystatus_kill_on_zone_map_exhaustion(pid_t pid);

/*
 * Kill a single process due to compressor space shortage.
 */
extern bool memorystatus_kill_on_VM_compressor_space_shortage(bool async);

/*
 * Asynchronously kill a single process due to VM Pageout Starvation (i.e.
 * a "stuck" external pageout thread).
 */
extern void memorystatus_kill_on_vps_starvation(void);

/*
 * Synchronously kill a single process due to vnode exhaustion
 */
extern bool memorystatus_kill_on_vnode_exhaustion(void);

/*
 * Wake up the memorystatus thread so it can do async kills.
 * The memorystatus thread will keep killing until the system is
 * considered healthy.
 */
extern void memorystatus_thread_wake(void);

/*
 * Respond to compressor exhaustion by waking the jetsam thread or
 * synchronously invoking a no-paging-space action.
 */
extern void memorystatus_respond_to_compressor_exhaustion(void);

/*
 * Respond to swap exhaustion by waking the jetsam thread or
 * synchronously invoking a no-paging-space action.
 */
extern void memorystatus_respond_to_swap_exhaustion(void);

__END_DECLS
