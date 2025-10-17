/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#ifndef _UAPI_SCC_H
#define _UAPI_SCC_H
#include <linux/sockios.h>
#define PA0HZP 0x00
#define EAGLE 0x01
#define PC100 0x02
#define PRIMUS 0x04
#define DRSI 0x08
#define BAYCOM 0x10
enum SCC_ioctl_cmds {
  SIOCSCCRESERVED = SIOCDEVPRIVATE,
  SIOCSCCCFG,
  SIOCSCCINI,
  SIOCSCCCHANINI,
  SIOCSCCSMEM,
  SIOCSCCGKISS,
  SIOCSCCSKISS,
  SIOCSCCGSTAT,
  SIOCSCCCAL
};
enum L1_params {
  PARAM_DATA,
  PARAM_TXDELAY,
  PARAM_PERSIST,
  PARAM_SLOTTIME,
  PARAM_TXTAIL,
  PARAM_FULLDUP,
  PARAM_SOFTDCD,
  PARAM_MUTE,
  PARAM_DTR,
  PARAM_RTS,
  PARAM_SPEED,
  PARAM_ENDDELAY,
  PARAM_GROUP,
  PARAM_IDLE,
  PARAM_MIN,
  PARAM_MAXKEY,
  PARAM_WAIT,
  PARAM_MAXDEFER,
  PARAM_TX,
  PARAM_HWEVENT = 31,
  PARAM_RETURN = 255
};
enum FULLDUP_modes {
  KISS_DUPLEX_HALF,
  KISS_DUPLEX_FULL,
  KISS_DUPLEX_LINK,
  KISS_DUPLEX_OPTIMA
};
#define TIMER_OFF 65535U
#define NO_SUCH_PARAM 65534U
enum HWEVENT_opts {
  HWEV_DCD_ON,
  HWEV_DCD_OFF,
  HWEV_ALL_SENT
};
#define RXGROUP 0100
#define TXGROUP 0200
enum CLOCK_sources {
  CLK_DPLL,
  CLK_EXTERNAL,
  CLK_DIVIDER,
  CLK_BRG
};
enum TX_state {
  TXS_IDLE,
  TXS_BUSY,
  TXS_ACTIVE,
  TXS_NEWFRAME,
  TXS_IDLE2,
  TXS_WAIT,
  TXS_TIMEOUT
};
typedef unsigned long io_port;
struct scc_stat {
  long rxints;
  long txints;
  long exints;
  long spints;
  long txframes;
  long rxframes;
  long rxerrs;
  long txerrs;
  unsigned int nospace;
  unsigned int rx_over;
  unsigned int tx_under;
  unsigned int tx_state;
  int tx_queued;
  unsigned int maxqueue;
  unsigned int bufsize;
};
struct scc_modem {
  long speed;
  char clocksrc;
  char nrz;
};
struct scc_kiss_cmd {
  int command;
  unsigned param;
};
struct scc_hw_config {
  io_port data_a;
  io_port ctrl_a;
  io_port data_b;
  io_port ctrl_b;
  io_port vector_latch;
  io_port special;
  int irq;
  long clock;
  char option;
  char brand;
  char escc;
};
struct scc_mem_config {
  unsigned int dummy;
  unsigned int bufsize;
};
struct scc_calibrate {
  unsigned int time;
  unsigned char pattern;
};
#endif
