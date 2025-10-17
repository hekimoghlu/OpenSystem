/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
#ifdef KERNEL_PRIVATE
#ifndef _I386_MACHINE_CHECK_H_
#define _I386_MACHINE_CHECK_H_

#include <stdint.h>

#include <i386/cpu_data.h>

/*
 * This header defines the machine check architecture for Pentium4 and Xeon.
 */

/*
 * Macro BITS(n,m) returns the number of bits between bit(n) and bit(m),
 * where (n>m). Macro BIT1(n) is cosmetic and returns 1.
 */
#define BITS(n, m)       ((n)-(m)+1)
#define BIT1(n)         (1)

/*
 * IA32 SDM 14.3.1 Machine-Check Global Control MSRs:
 */
#define IA32_MCG_CAP            (0x179)
typedef union {
	struct {
		uint64_t        count                   :BITS(7, 0);
		uint64_t        mcg_ctl_p               :BIT1(8);
		uint64_t        mcg_ext_p               :BIT1(9);
		uint64_t        mcg_ext_corr_err_p      :BIT1(10);
		uint64_t        mcg_tes_p               :BIT1(11);
		uint64_t        mcg_ecms                :BIT1(12);
		uint64_t        mcg_reserved2           :BITS(15, 13);
		uint64_t        mcg_ext_cnt             :BITS(23, 16);
		uint64_t        mcg_ser_p               :BIT1(24);
	}          bits;
	uint64_t   u64;
} ia32_mcg_cap_t;

#define IA32_MCG_STATUS         (0x17A)
typedef union {
	struct {
		uint64_t        ripv                    :BIT1(0);
		uint64_t        eipv                    :BIT1(1);
		uint64_t        mcip                    :BIT1(2);
	}          bits;
	uint64_t   u64;
} ia32_mcg_status_t;

#define IA32_MCG_CTL            (0x17B)
typedef uint64_t        ia32_mcg_ctl_t;
#define IA32_MCG_CTL_ENABLE     (0xFFFFFFFFFFFFFFFFULL)
#define IA32_MCG_CTL_DISABLE    (0x0ULL)


/*
 * IA32 SDM 14.3.2 Error-Reporting Register Banks:
 */
#define IA32_MCi_CTL(i)         (0x400 + 4*(i))
#define IA32_MCi_STATUS(i)      (0x401 + 4*(i))
#define IA32_MCi_ADDR(i)        (0x402 + 4*(i))
#define IA32_MCi_MISC(i)        (0x403 + 4*(i))

#define IA32_MC0_CTL            IA32_MCi_CTL(0)
#define IA32_MC0_STATUS         IA32_MCi_STATUS(0)
#define IA32_MC0_ADDR           IA32_MCi_ADDR(0)
#define IA32_MC0_MISC           IA32_MCi_MISC(0)

#define IA32_MC1_CTL            IA32_MCi_CTL(1)
#define IA32_MC1_STATUS         IA32_MCi_STATUS(1)
#define IA32_MC1_ADDR           IA32_MCi_ADDR(1)
#define IA32_MC1_MISC           IA32_MCi_MISC(1)

#define IA32_MC2_CTL            IA32_MCi_CTL(2)
#define IA32_MC2_STATUS         IA32_MCi_STATUS(2)
#define IA32_MC2_ADDR           IA32_MCi_ADDR(2)
#define IA32_MC2_MISC           IA32_MCi_MISC(2)

#define IA32_MC3_CTL            IA32_MCi_CTL(3)
#define IA32_MC3_STATUS         IA32_MCi_STATUS(3)
#define IA32_MC3_ADDR           IA32_MCi_ADDR(3)
#define IA32_MC3_MISC           IA32_MCi_MISC(3)

#define IA32_MC4_CTL            IA32_MCi_CTL(4)
#define IA32_MC4_STATUS         IA32_MCi_STATUS(4)
#define IA32_MC4_ADDR           IA32_MCi_ADDR(4)
#define IA32_MC4_MISC           IA32_MCi_MISC(4)

typedef uint64_t        ia32_mci_ctl_t;
#define IA32_MCi_CTL_EE(j)      (0x1ULL << (j))
#define IA32_MCi_CTL_ENABLE_ALL (0xFFFFFFFFFFFFFFFFULL)

typedef union {
	struct {
		uint64_t        mca_error               :BITS(15, 0);
		uint64_t        model_specific_error    :BITS(31, 16);
		uint64_t        other_information       :BITS(56, 32);
		uint64_t        pcc                     :BIT1(57);
		uint64_t        addrv                   :BIT1(58);
		uint64_t        miscv                   :BIT1(59);
		uint64_t        en                      :BIT1(60);
		uint64_t        uc                      :BIT1(61);
		uint64_t        over                    :BIT1(62);
		uint64_t        val                     :BIT1(63);
	}           bits;
	struct {        /* Variant if threshold-based error status present: */
		uint64_t        mca_error               :BITS(15, 0);
		uint64_t        model_specific_error    :BITS(31, 16);
		uint64_t        other_information       :BITS(52, 32);
		uint64_t        threshold               :BITS(54, 53);
		uint64_t        ar                      :BIT1(55);
		uint64_t        s                       :BIT1(56);
		uint64_t        pcc                     :BIT1(57);
		uint64_t        addrv                   :BIT1(58);
		uint64_t        miscv                   :BIT1(59);
		uint64_t        en                      :BIT1(60);
		uint64_t        uc                      :BIT1(61);
		uint64_t        over                    :BIT1(62);
		uint64_t        val                     :BIT1(63);
	}           bits_tes_p;
	uint64_t    u64;
} ia32_mci_status_t;

/* Values for threshold_status if mcg_tes_p == 1 and uc == 0 */
#define THRESHOLD_STATUS_NO_TRACKING    0
#define THRESHOLD_STATUS_GREEN          1
#define THRESHOLD_STATUS_YELLOW         2
#define THRESHOLD_STATUS_RESERVED       3

typedef uint64_t        ia32_mci_addr_t;
typedef uint64_t        ia32_mci_misc_t;

extern void             mca_cpu_alloc(cpu_data_t *cdp);
extern void             mca_cpu_init(void);
extern void             mca_dump(void);
extern void             mca_check_save(void);
extern boolean_t        mca_is_cmci_present(void);

#endif  /* _I386_MACHINE_CHECK_H_ */
#endif  /* KERNEL_PRIVATE */
