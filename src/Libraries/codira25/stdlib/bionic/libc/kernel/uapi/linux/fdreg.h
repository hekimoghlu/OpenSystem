/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#ifndef _LINUX_FDREG_H
#define _LINUX_FDREG_H
#define FD_SRA 0
#define FD_SRB 1
#define FD_DOR 2
#define FD_TDR 3
#define FD_DSR 4
#define FD_STATUS 4
#define FD_DATA 5
#define FD_DIR 7
#define FD_DCR 7
#define STATUS_BUSYMASK 0x0F
#define STATUS_BUSY 0x10
#define STATUS_DMA 0x20
#define STATUS_DIR 0x40
#define STATUS_READY 0x80
#define ST0_DS 0x03
#define ST0_HA 0x04
#define ST0_NR 0x08
#define ST0_ECE 0x10
#define ST0_SE 0x20
#define ST0_INTR 0xC0
#define ST1_MAM 0x01
#define ST1_WP 0x02
#define ST1_ND 0x04
#define ST1_OR 0x10
#define ST1_CRC 0x20
#define ST1_EOC 0x80
#define ST2_MAM 0x01
#define ST2_BC 0x02
#define ST2_SNS 0x04
#define ST2_SEH 0x08
#define ST2_WC 0x10
#define ST2_CRC 0x20
#define ST2_CM 0x40
#define ST3_HA 0x04
#define ST3_DS 0x08
#define ST3_TZ 0x10
#define ST3_RY 0x20
#define ST3_WP 0x40
#define ST3_FT 0x80
#define FD_RECALIBRATE 0x07
#define FD_SEEK 0x0F
#define FD_READ 0xE6
#define FD_WRITE 0xC5
#define FD_SENSEI 0x08
#define FD_SPECIFY 0x03
#define FD_FORMAT 0x4D
#define FD_VERSION 0x10
#define FD_CONFIGURE 0x13
#define FD_PERPENDICULAR 0x12
#define FD_GETSTATUS 0x04
#define FD_DUMPREGS 0x0E
#define FD_READID 0xEA
#define FD_UNLOCK 0x14
#define FD_LOCK 0x94
#define FD_RSEEK_OUT 0x8f
#define FD_RSEEK_IN 0xcf
#define FD_PARTID 0x18
#define FD_SAVE 0x2e
#define FD_DRIVESPEC 0x8e
#define FD_RESTORE 0x4e
#define FD_POWERDOWN 0x27
#define FD_FORMAT_N_WRITE 0xef
#define FD_OPTION 0x33
#define DMA_READ 0x46
#define DMA_WRITE 0x4A
#define FDC_NONE 0x00
#define FDC_UNKNOWN 0x10
#define FDC_8272A 0x20
#define FDC_765ED 0x30
#define FDC_82072 0x40
#define FDC_82072A 0x45
#define FDC_82077_ORIG 0x51
#define FDC_82077 0x52
#define FDC_82078_UNKN 0x5f
#define FDC_82078 0x60
#define FDC_82078_1 0x61
#define FDC_S82078B 0x62
#define FDC_87306 0x63
#define FD_RESET_DELAY 20
#endif
