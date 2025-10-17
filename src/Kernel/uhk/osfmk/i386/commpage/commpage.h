/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#ifndef _I386_COMMPAGE_H
#define _I386_COMMPAGE_H

#ifndef __ASSEMBLER__
#include <stdint.h>
#include <mach/boolean.h>
#include <mach/vm_types.h>
#include <machine/cpu_capabilities.h>
#endif /* __ASSEMBLER__ */

/* The following macro is used to generate the 64-bit commpage address for a given
 * routine, based on its 32-bit address.  This is used in the kernel to compile
 * the 64-bit commpage.  Since the kernel can be a 32-bit object, cpu_capabilities.h
 * only defines the 32-bit address.
 */
#define _COMM_PAGE_32_TO_64( ADDRESS )  ( ADDRESS + _COMM_PAGE64_START_ADDRESS - _COMM_PAGE32_START_ADDRESS )


#ifdef  __ASSEMBLER__

#define COMMPAGE_DESCRIPTOR_NAME(label)  _commpage_ ## label

#define COMMPAGE_DESCRIPTOR_FIELD_POINTER .quad
#define COMMPAGE_DESCRIPTOR_REFERENCE(label) \
	.quad COMMPAGE_DESCRIPTOR_NAME(label)

#define COMMPAGE_FUNCTION_START(label, codetype, alignment) \
.text								;\
.code ## codetype						;\
.align alignment, 0x90						;\
L ## label ## :

#define COMMPAGE_DESCRIPTOR(label, address)                      \
L ## label ## _end:						;\
.set L ## label ## _size, L ## label ## _end - L ## label	;\
.const_data							;\
.private_extern COMMPAGE_DESCRIPTOR_NAME(label)			;\
COMMPAGE_DESCRIPTOR_NAME(label) ## :				;\
    COMMPAGE_DESCRIPTOR_FIELD_POINTER	L ## label              ;\
    .long				L ## label ## _size	;\
    .long				address			;\
.text


/* COMMPAGE_CALL(target,from,start)
 *
 * This macro compiles a relative near call to one
 * commpage routine from another.
 * The assembler cannot handle this directly because the code
 * is not being assembled at the address at which it will execute.
 * The alternative to this macro would be to use an
 * indirect call, which is slower because the target of an
 * indirect branch is poorly predicted.
 * The macro arguments are:
 *	target = the commpage routine we are calling
 *	from   = the commpage routine we are in now
 *	start  = the label at the start of the code for this func
 * This is admitedly ugly and fragile.  Is there a better way?
 */
#define COMMPAGE_CALL(target, from, start)                        \
	COMMPAGE_CALL_INTERNAL(target,from,start,__LINE__)

#define COMMPAGE_CALL_INTERNAL(target, from, start, unique)        \
	.byte 0xe8						;\
.set UNIQUEID(unique), L ## start - . + target - from - 4	;\
	.long	UNIQUEID(unique)

#define UNIQUEID(name)  L ## name

/* COMMPAGE_JMP(target,from,start)
 *
 * This macro perform a jump to another commpage routine.
 * Used to return from the PFZ by jumping via a return outside the PFZ.
 */
#define COMMPAGE_JMP(target, from, start)                         \
	jmp      L ## start - from + target

#else /* __ASSEMBLER__ */

/* Each potential commpage routine is described by one of these.
 * Note that the COMMPAGE_DESCRIPTOR macro (above), used in
 * assembly language, must agree with this.
 */

typedef struct  commpage_descriptor     {
	void                *code_address;                      // address of code
	uint32_t            code_length;                        // length in bytes
	uint32_t            commpage_address;                   // put at this address (_COMM_PAGE_BCOPY etc)
} commpage_descriptor;


/* Warning: following structure must match the layout of the commpage.  */
/* This is the data starting at _COMM_PAGE_TIME_DATA_START, ie for nanotime() and gettimeofday() */

typedef volatile struct commpage_time_data      {
	uint64_t        nt_tsc_base;                            // _COMM_PAGE_NT_TSC_BASE
	uint32_t        nt_scale;                               // _COMM_PAGE_NT_SCALE
	uint32_t        nt_shift;                               // _COMM_PAGE_NT_SHIFT
	uint64_t        nt_ns_base;                             // _COMM_PAGE_NT_NS_BASE
	uint32_t        nt_generation;                          // _COMM_PAGE_NT_GENERATION
	uint32_t        gtod_generation;                        // _COMM_PAGE_GTOD_GENERATION
	uint64_t        gtod_ns_base;                           // _COMM_PAGE_GTOD_NS_BASE
	uint64_t        gtod_sec_base;                          // _COMM_PAGE_GTOD_SEC_BASE
} commpage_time_data;

extern  char    *commPagePtr32;                         // virt address of 32-bit commpage in kernel map
extern  char    *commPagePtr64;                         // ...and of 64-bit commpage

extern  void    commpage_set_timestamp(uint64_t abstime, uint64_t sec, uint64_t frac, uint64_t scale, uint64_t tick_per_sec);
#define commpage_disable_timestamp() commpage_set_timestamp( 0, 0, 0, 0, 0 );
extern  void    commpage_set_nanotime(uint64_t tsc_base, uint64_t ns_base, uint32_t scale, uint32_t shift);
extern  void    commpage_set_memory_pressure(unsigned int  pressure);
extern  void    commpage_set_spin_count(unsigned int  count);
extern  void    commpage_sched_gen_inc(void);
extern  void    commpage_update_active_cpus(void);
extern  void    commpage_update_mach_approximate_time(uint64_t abstime);
extern  void    commpage_update_mach_continuous_time(uint64_t sleeptime);
extern  void    commpage_update_boottime(uint64_t boottime_usec);
extern  void    commpage_update_kdebug_state(void);
extern  void    commpage_update_atm_diagnostic_config(uint32_t);
extern  void    commpage_update_dof(boolean_t enabled);
extern  void    commpage_update_dyld_flags(uint64_t value);
extern  void    commpage_post_ucode_update(void);

extern  uint32_t        commpage_is_in_pfz32(uint32_t);
extern  uint32_t        commpage_is_in_pfz64(addr64_t);

#endif  /* __ASSEMBLER__ */

#endif /* _I386_COMMPAGE_H */
