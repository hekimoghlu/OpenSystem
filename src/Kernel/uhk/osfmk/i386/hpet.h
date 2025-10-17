/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#ifndef _I386_HPET_H_
#define _I386_HPET_H_

/*
 * HPET kernel functions to support the HPET KEXT and the
 * power management KEXT.
 */


/*
 *	Memory mapped registers for the HPET
 */
typedef struct hpetReg {
	uint64_t        GCAP_ID;                /* General capabilities */
	uint64_t        rsv1;
	uint64_t        GEN_CONF;               /* General configuration */
	uint64_t        rsv2;
	uint64_t        GINTR_STA;              /* General Interrupt status */
	uint64_t        rsv3[25];
	uint64_t        MAIN_CNT;               /* Main counter */
	uint64_t        rsv4;
	uint64_t        TIM0_CONF;              /* Timer 0 config and cap */
#define                 TIM_CONF 0
#define                 Tn_INT_ENB_CNF 4
	uint64_t        TIM0_COMP;              /* Timer 0 comparator */
#define                 TIM_COMP 8
	uint64_t        rsv5[2];
	uint64_t        TIM1_CONF;              /* Timer 1 config and cap */
	uint64_t        TIM1_COMP;              /* Timer 1 comparator */
	uint64_t        rsv6[2];
	uint64_t        TIM2_CONF;              /* Timer 2 config and cap */
	uint64_t        TIM2_COMP;              /* Timer 2 comparator */
	uint64_t        rsv7[2];
} hpetReg;
typedef struct  hpetReg hpetReg_t;

typedef struct hpetTimer {
	uint64_t        Config;         /* Timer config and capabilities */
	uint64_t        Compare;        /* Timer comparitor */
} hpetTimer_t;

struct hpetInfo {
	uint64_t        hpetCvtt2n;
	uint64_t        hpetCvtn2t;
	uint64_t        tsc2hpet;
	uint64_t        hpet2tsc;
	uint64_t        bus2hpet;
	uint64_t        hpet2bus;
	uint32_t        rcbaArea;
	uint32_t        rcbaAreap;
};
typedef struct hpetInfo hpetInfo_t;

struct hpetRequest {
	uint32_t        flags;
	uint32_t        hpetOffset;
	uint32_t        hpetVector;
};
typedef struct hpetRequest hpetRequest_t;

#define HPET_REQFL_64BIT        0x00000001      /* Timer is 64 bits */

extern uint64_t hpetFemto;
extern uint64_t hpetFreq;
extern uint64_t hpetCvtt2n;
extern uint64_t hpetCvtn2t;
extern uint64_t tsc2hpet;
extern uint64_t hpet2tsc;
extern uint64_t bus2hpet;
extern uint64_t hpet2bus;

extern vm_offset_t rcbaArea;
extern uint32_t rcbaAreap;

extern void map_rcbaAread(void);
extern void hpet_init(void);

extern void hpet_save(void);
extern void hpet_restore(void);

#ifdef XNU_KERNEL_PRIVATE
extern int HPETInterrupt(void);
#endif

extern int hpet_register_callback(int (*hpet_reqst)(uint32_t apicid, void *arg, hpetRequest_t *hpet), void *arg);
extern int hpet_request(uint32_t cpu);

extern uint64_t rdHPET(void);
extern void hpet_get_info(hpetInfo_t *info);

#define hpetAddr        0xFED00000
#define hptcAE          0x80

#endif  /* _I386_HPET_H_ */

#endif  /* KERNEL_PRIVATE */
