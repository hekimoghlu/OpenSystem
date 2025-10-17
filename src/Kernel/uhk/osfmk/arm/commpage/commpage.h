/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#ifndef _ARM_COMMPAGE_H
#define _ARM_COMMPAGE_H

#ifndef __ASSEMBLER__
#include <stdint.h>
#include <mach/vm_types.h>
#endif /* __ASSEMBLER__ */

extern void     commpage_set_timestamp(uint64_t tbr, uint64_t secs, uint64_t frac, uint64_t scale, uint64_t tick_per_sec);
#define commpage_disable_timestamp() commpage_set_timestamp( 0, 0, 0, 0, 0 );
extern  void    commpage_set_memory_pressure( unsigned int pressure );
extern  void    commpage_update_active_cpus(void);
extern  void    commpage_set_spin_count(unsigned int  count);
extern  void    commpage_update_timebase(void);
extern  void    commpage_update_mach_approximate_time(uint64_t);
extern  void    commpage_update_kdebug_state(void);
extern  void    commpage_update_atm_diagnostic_config(uint32_t);
extern  void    commpage_update_mach_continuous_time(uint64_t sleeptime);
extern  void    commpage_update_mach_continuous_time_hw_offset(uint64_t offset);
extern  void    commpage_update_multiuser_config(uint32_t);
extern  void    commpage_update_boottime(uint64_t boottime_usec);
extern  void    commpage_set_remotetime_params(double rate, uint64_t base_local_ts, uint64_t base_remote_ts);
extern  void    commpage_update_dof(boolean_t enabled);
extern  void    commpage_update_dyld_flags(uint64_t value);
extern uint32_t commpage_is_in_pfz64(addr64_t addr);
extern  void    commpage_update_apt_active(bool active);

#endif /* _ARM_COMMPAGE_H */
