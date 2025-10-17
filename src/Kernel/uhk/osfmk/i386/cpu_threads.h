/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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
#ifndef _I386_CPU_THREADS_H_
#define _I386_CPU_THREADS_H_

#include <i386/cpu_data.h>
#include <i386/cpu_topology.h>
#include <i386/mp.h>

/*
 * These are defined here rather than in cpu_topology.h so as to keep
 * cpu_topology.h from having a dependency on cpu_data.h.
 */
#define CPU_THREAD_MASK                 0x00000001
#define cpu_to_core_lapic(cpu)          (cpu_to_lapic[cpu] & ~CPU_THREAD_MASK)
#define cpu_to_core_cpu(cpu)            (lapic_to_cpu[cpu_to_core_lapic(cpu)])
#define cpu_to_logical_cpu(cpu)         (cpu_to_lapic[cpu] & CPU_THREAD_MASK)
#define cpu_is_core_cpu(cpu)            (cpu_to_logical_cpu(cpu) == 0)

#define _cpu_to_lcpu(cpu)               (&cpu_datap(cpu)->lcpu)
#define _cpu_to_core(cpu)               (_cpu_to_lcpu(cpu)->core)
#define _cpu_to_package(cpu)            (_cpu_to_core(cpu)->package)

#define cpu_to_lcpu(cpu)                ((cpu_datap(cpu) != NULL) ? _cpu_to_lcpu(cpu) : NULL)
#define cpu_to_core(cpu)                ((cpu_to_lcpu(cpu) != NULL) ? _cpu_to_lcpu(cpu)->core : NULL)
#define cpu_to_die(cpu)                 ((cpu_to_lcpu(cpu) != NULL) ? _cpu_to_lcpu(cpu)->die : NULL)
#define cpu_to_package(cpu)             ((cpu_to_lcpu(cpu) != NULL) ? _cpu_to_lcpu(cpu)->package : NULL)

/* Fast access: */
#define x86_lcpu()                      (&current_cpu_datap()->lcpu)
#define x86_core()                      (x86_lcpu()->core)
#define x86_die()                       (x86_lcpu()->die)
#define x86_package()                   (x86_lcpu()->package)

#define cpu_is_same_core(cpu1, cpu2)     (cpu_to_core(cpu1) == cpu_to_core(cpu2))
#define cpu_is_same_die(cpu1, cpu2)      (cpu_to_die(cpu1) == cpu_to_die(cpu2))
#define cpu_is_same_package(cpu1, cpu2)  (cpu_to_package(cpu1) == cpu_to_package(cpu2))
#define cpus_share_cache(cpu1, cpu2, _cl) (cpu_to_lcpu(cpu1)->caches[_cl] == cpu_to_lcpu(cpu2)->caches[_cl])

/* always take the x86_topo_lock with mp_safe_spin_lock */
boolean_t       mp_safe_spin_lock(usimple_lock_t lock);
extern decl_simple_lock_data(, x86_topo_lock);

extern void *cpu_thread_alloc(int);
extern void cpu_thread_init(void);
extern void cpu_thread_halt(void);

extern void x86_set_logical_topology(x86_lcpu_t *lcpu, int pnum, int lnum);
extern void x86_validate_topology(void);

extern x86_topology_parameters_t        topoParms;

extern boolean_t        topo_dbg;
#define TOPO_DBG(x...)                  \
	do {                            \
	        if (topo_dbg)           \
	                kprintf(x);     \
	} while (0)                     \

#endif /* _I386_CPU_THREADS_H_ */
