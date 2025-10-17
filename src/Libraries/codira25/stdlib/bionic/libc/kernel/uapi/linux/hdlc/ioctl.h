/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#ifndef __HDLC_IOCTL_H__
#define __HDLC_IOCTL_H__
#define GENERIC_HDLC_VERSION 4
#define CLOCK_DEFAULT 0
#define CLOCK_EXT 1
#define CLOCK_INT 2
#define CLOCK_TXINT 3
#define CLOCK_TXFROMRX 4
#define ENCODING_DEFAULT 0
#define ENCODING_NRZ 1
#define ENCODING_NRZI 2
#define ENCODING_FM_MARK 3
#define ENCODING_FM_SPACE 4
#define ENCODING_MANCHESTER 5
#define PARITY_DEFAULT 0
#define PARITY_NONE 1
#define PARITY_CRC16_PR0 2
#define PARITY_CRC16_PR1 3
#define PARITY_CRC16_PR0_CCITT 4
#define PARITY_CRC16_PR1_CCITT 5
#define PARITY_CRC32_PR0_CCITT 6
#define PARITY_CRC32_PR1_CCITT 7
#define LMI_DEFAULT 0
#define LMI_NONE 1
#define LMI_ANSI 2
#define LMI_CCITT 3
#define LMI_CISCO 4
#ifndef __ASSEMBLY__
typedef struct {
  unsigned int clock_rate;
  unsigned int clock_type;
  unsigned short loopback;
} sync_serial_settings;
typedef struct {
  unsigned int clock_rate;
  unsigned int clock_type;
  unsigned short loopback;
  unsigned int slot_map;
} te1_settings;
typedef struct {
  unsigned short encoding;
  unsigned short parity;
} raw_hdlc_proto;
typedef struct {
  unsigned int t391;
  unsigned int t392;
  unsigned int n391;
  unsigned int n392;
  unsigned int n393;
  unsigned short lmi;
  unsigned short dce;
} fr_proto;
typedef struct {
  unsigned int dlci;
} fr_proto_pvc;
typedef struct {
  unsigned int dlci;
  char master[IFNAMSIZ];
} fr_proto_pvc_info;
typedef struct {
  unsigned int interval;
  unsigned int timeout;
} cisco_proto;
typedef struct {
  unsigned short dce;
  unsigned int modulo;
  unsigned int window;
  unsigned int t1;
  unsigned int t2;
  unsigned int n2;
} x25_hdlc_proto;
#endif
#endif
