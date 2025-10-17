/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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
#ifndef _I386_X86_HYPERCALL_H_
#define _I386_X86_HYPERCALL_H_

#include <kern/hvg_hypercall.h>
#include <i386/cpuid.h>

/*
 * Apple Hypercall Calling Convention (x64)
 *
 * Registers |                Usage                       |
 * --------------------------------------------------------
 *      %rax |    In:  hypercall code                     |
 *           |    Out: if RFLAGS.CF = 0 (success)         |
 *           |           hypercall output[0]              |
 *           |         if RFLAGS.CF = 1 (error)           |
 *           |           hypercall error value            |
 *      %rdi |    In:  1st argument                       |
 *           |    Out: hypercall output[1]                |
 *      %rsi |    In:  2nd argument                       |
 *           |    Out: hypercall output[2]                |
 *      %rdx |    In:  3rd argument                       |
 *           |    Out: hypercall output[3]                |
 *      %rcx |    In:  4th argument                       |
 *           |    Out: hypercall output[4]                |
 *      %r8  |    In:  5th argument                       |
 *           |    Out: hypercall output[5]                |
 *      %r9  |    In:  6th argument                       |
 *           |    Out: hypercall output[6]                |
 *
 * %rax is used by the caller to specify hypercall code. When a hypercall fails,
 * the hypervisor stores errno in %rax. A successful hypercall returns the
 * output of the call in %rax, %rdi, %rsi, %rdx, %rcx, %r8, and %r9.
 */

typedef struct hvg_hcall_output_regs {
	uint64_t   rax;
	uint64_t   rdi;
	uint64_t   rsi;
	uint64_t   rdx;
	uint64_t   rcx;
	uint64_t   r8;
	uint64_t   r9;
} hvg_hcall_output_regs_t;

/*
 * To avoid collision with other hypercall interfaces (e.g., KVM) in the vmcall
 * namespace, Apple hypercalls put "A" (0x41) in the top byte of %eax so that
 * hypervisors can support multiple hypercall interfaces simultaneously and
 * handle Apple hypercalls correctly for compatiblity.
 *
 * For example, KVM uses the same vmcall instruction and has call code 1 for
 * KVM_HC_VAPIC_POLL_IRQ. When invoking an Apple hypercall with code 1, a
 * hypervisor will not accidentially treat the Apple hypercall as a KVM call.
 */

#define HVG_HCALL_CODE(code) ('A' << 24 | (code & 0xFFFFFF))

/*
 * Caller is responsible for checking the existence of Apple Hypercall
 * before invoking Apple hypercalls.
 */

#if defined(MACH_KERNEL_PRIVATE)

#define HVG_HCALL_RETURN(rax) {\
	__asm__ __volatile__ goto (\
	                           "jnc 2f  \n\t" \
	                           "jmp %l0 \n\t" \
	                           "2:      \n\t" \
	                          : /* no output */ \
	                          : /* no input */  \
	                          : /* no clobber */ \
	                          : error);\
	return HVG_HCALL_SUCCESS;\
error:\
	return (hvg_hcall_return_t)rax;\
}

static hvg_hcall_return_t
hvg_hypercall6(uint64_t code, uint64_t rdi, uint64_t rsi, uint64_t rdx, uint64_t rcx, uint64_t r8, uint64_t r9,
    hvg_hcall_output_regs_t *output)
{
	__asm__ __volatile__ ("movq %12, %%r8  \n\t"
                          "movq %13, %%r9  \n\t"
                          "vmcall          \n\t"
                          "movq %%r8, %5   \n\t"
                          "movq %%r9, %6   \n\t"
                        : "=a" (output->rax),         /* %0:  output[0] */
                          "=D" (output->rdi),         /* %1:  output[1] */
                          "=S" (output->rsi),         /* %2:  output[2] */
                          "=d" (output->rdx),         /* %3:  output[3] */
                          "=c" (output->rcx),         /* %4:  output[4] */
                          "=r" (output->r8),          /* %5:  output[5] */
                          "=r" (output->r9)           /* %6:  output[6] */
                        : "a"  (HVG_HCALL_CODE(code)),/* %7:  call code */
                          "D"  (rdi),                 /* %8:  arg[0]    */
                          "S"  (rsi),                 /* %9:  arg[1]    */
                          "d"  (rdx),                 /* %10: arg[2]    */
                          "c"  (rcx),                 /* %11: arg[3]    */
                          "r"  (r8),                  /* %12: arg[4]    */
                          "r"  (r9)                   /* %13: arg[5]    */
                        : "memory", "r8", "r9");
	HVG_HCALL_RETURN(output->rax);
}

static inline hvg_hcall_return_t
hvg_hypercall0(const uint64_t code,
    hvg_hcall_output_regs_t *output)
{
	return hvg_hypercall6(code, 0, 0, 0, 0, 0, 0, output);
}

static inline hvg_hcall_return_t
hvg_hypercall1(const uint64_t code,
    const uint64_t rdi,
    hvg_hcall_output_regs_t *output)
{
	return hvg_hypercall6(code, rdi, 0, 0, 0, 0, 0, output);
}

static inline hvg_hcall_return_t
hvg_hypercall2(const uint64_t code,
    const uint64_t rdi, const uint64_t rsi,
    hvg_hcall_output_regs_t *output)
{
	return hvg_hypercall6(code, rdi, rsi, 0, 0, 0, 0, output);
}

static inline hvg_hcall_return_t
hvg_hypercall3(const uint64_t code,
    const uint64_t rdi, const uint64_t rsi, const uint64_t rdx,
    hvg_hcall_output_regs_t *output)
{
	return hvg_hypercall6(code, rdi, rsi, rdx, 0, 0, 0, output);
}

static inline hvg_hcall_return_t
hvg_hypercall4(const uint64_t code,
    const uint64_t rdi, const uint64_t rsi, const uint64_t rdx, const uint64_t rcx,
    hvg_hcall_output_regs_t *output)
{
	return hvg_hypercall6(code, rdi, rsi, rdx, rcx, 0, 0, output);
}

static inline hvg_hcall_return_t
hvg_hypercall5(const uint64_t code,
    const uint64_t rdi, const uint64_t rsi, const uint64_t rdx, const uint64_t rcx, const uint64_t r8,
    hvg_hcall_output_regs_t *output)
{
	return hvg_hypercall6(code, rdi, rsi, rdx, rcx, r8, 0, output);
}

static inline long
kvmcompat_hypercall2(unsigned long code, unsigned long a0, unsigned long a1)
{
	long retval;
	__asm__ __volatile__ (
		"vmcall"
		: "=a" (retval)
		: "a" (code), "b" (a0), "c" (a1)
		: "memory"
	);
	return retval;
}
#endif /* MACH_KERNEL_PRIVATE */
#endif /* _I386_X86_HYPERCALL_H_ */
