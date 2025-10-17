/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#include <signal.h>

#define MAXRESETRETRANSMITS (3)
/*#define INSESSION(p, src, sport, dst, dport)			\
		(((p)->ip.ip_src == (src)) && ((p)->ip.ip_dst == (dst)) &&	\
		 ((p)->ip.ip_p == IPPROTOCOL_TCP) &&			\
		 ((p)->tcp.tcp_sport == htons(sport)) &&			\
		 ((p)->tcp.tcp_dport == htons(dport)))*/

#define INSESSION(p, src, sport, dst, dport)			\
		(((p)->ip->ip_src == (src)) && ((p)->ip->ip_dst == (dst)) &&	\
		 ((p)->ip->ip_p == IPPROTOCOL_TCP) &&			\
		 ((p)->tcp->tcp_sport == htons(sport)) &&			\
		 ((p)->tcp->tcp_dport == htons(dport)))

#define SEQ_LT(a,b) ((int)((a)-(b)) < 0)
#define SEQ_LEQ(a,b) ((int)((a)-(b)) <= 0)
#define SEQ_GT(a,b) ((int)((a)-(b)) > 0)
#define SEQ_GEQ(a,b) ((int)((a)-(b)) >= 0)

#define DEFAULT_TARGETPORT  (80)
#define DEFAULT_MSS	1360
#define DEFAULT_MTU 1500
#define	RTT_TO_MULT	5
#define PLOTDIFF 0.00009

/* Response codes */
#define  FAIL                        -1
#define  SUCCESS                      0
#define  NO_TARGET_CANON_INFO         1
#define  NO_LOCAL_HOSTNAME            2
#define  NO_SRC_CANON_INFO            3
#define  NO_SESSION_ESTABLISH         4
#define  MSS_TOO_SMALL                5
#define  BAD_ARGS                     6
#define  FIREWALL_ERR                 7
#define  ERR_SOCKET_OPEN              8
#define  ERR_SOCKOPT                  9
#define  ERR_MEM_ALLOC               10
#define  NO_CONNECTION               11
#define  MSS_ERR                     12
#define  BUFFER_OVERFLOW             13
#define  UNWANTED_PKT_DROP           14
#define  EARLY_RST                   15
#define  UNEXPECTED_PKT              16
#define  DIFF_FLOW                   17
#define  ERR_CHECKSUM                18
#define  NOT_ENOUGH_PKTS             19
#define  BAD_OPT_LEN                 20
#define  TOO_MANY_PKTS               21
#define  NO_DATA_RCVD                22
#define  NO_TRGET_SPECIFIED          23
#define  BAD_OPTIONS                 24
#define  TOO_MANY_TIMEOUTS           25
#define  TOO_MANY_RXMTS              26
#define  NO_SACK                     27
#define  ERR_IN_SB_CALC              28
#define  TOO_MANY_HOLES              29
#define  TOO_MANY_DROPS              30
#define  UNWANTED_PKT_REORDER        31
#define  NO_PMTUD_ENABLED            32
#define  UNKNOWN_BEHAVIOR            33
#define  NO_SYNACK_RCVD              34
#define  SEND_REQUEST_FAILED         35
#define  PKT_SIZE_CHANGED            36
#define	 ECN_SYN_DROP                37

#define DEFAULT_FILENAME "/"

#define RTT_TO_MULT 5
#define SYNTIMEOUT    (2.0)
#define REXMITDELAY   (2.0)
#define MAXSYNRETRANSMITS  (6)
#define MAXDATARETRANSMITS  (6)

/* HTTP Response Codes */
#define HTTP_OK                     "200"


void SendReset(); 
void SigHandle (int signo);
void Cleanup(); 
void Quit(int how);
double GetTime(); 
double GetTimeMicroSeconds(); 
void PrintTimeStamp(struct timeval *ts); 
void processBadPacket (struct IPPacket *p);
void busy_wait (double wait);
