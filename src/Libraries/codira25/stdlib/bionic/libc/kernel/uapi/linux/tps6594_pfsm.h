/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#ifndef __TPS6594_PFSM_H
#define __TPS6594_PFSM_H
#include <linux/const.h>
#include <linux/ioctl.h>
#include <linux/types.h>
struct pmic_state_opt {
  __u8 gpio_retention;
  __u8 ddr_retention;
  __u8 mcu_only_startup_dest;
};
#define PMIC_BASE 'P'
#define PMIC_GOTO_STANDBY _IO(PMIC_BASE, 0)
#define PMIC_GOTO_LP_STANDBY _IO(PMIC_BASE, 1)
#define PMIC_UPDATE_PGM _IO(PMIC_BASE, 2)
#define PMIC_SET_ACTIVE_STATE _IO(PMIC_BASE, 3)
#define PMIC_SET_MCU_ONLY_STATE _IOW(PMIC_BASE, 4, struct pmic_state_opt)
#define PMIC_SET_RETENTION_STATE _IOW(PMIC_BASE, 5, struct pmic_state_opt)
#endif
