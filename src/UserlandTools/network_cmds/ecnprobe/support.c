/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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
#include <sys/types.h>
#include <sys/param.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "base.h"
#include "inet.h"
#include "session.h"
#include "capture.h"
#include "support.h"

extern struct TcpSession session; 

void SendReset()
{
  struct IPPacket *p;
  int i;

  if (session.dont_send_reset)
	  return;

  if ((p = (struct IPPacket *)calloc(1, sizeof(struct IPPacket))) == NULL) {
    perror("ERROR: Could not allocate RST packet:") ;
    Quit(ERR_MEM_ALLOC) ; 
  }

  if ((p->ip = (struct IpHeader *)calloc(1, sizeof(struct IpHeader))) == NULL) {
    perror("ERROR: Could not allocate IP Header for RST packet:") ;
    Quit(ERR_MEM_ALLOC) ; 
  }

  if ((p->tcp = (struct TcpHeader *)calloc(1, sizeof(struct TcpHeader))) == NULL) {
    perror("ERROR: Could not allocate TCP Header for RST packet:") ;
    Quit(ERR_MEM_ALLOC) ; 
  }
  
  for (i = 0; i < MAXRESETRETRANSMITS; i++) {
    SendSessionPacket(p, 
		      //sizeof(struct IPPacket), 
		      sizeof(struct IpHeader) + sizeof(struct TcpHeader),
		      TCPFLAGS_RST, 
		      0,
		      0, 
		      0);
  }

/*  free(p->ip);
  free(p->tcp);
  free(p);
*/

}

double GetTime()
{
  struct timeval tv;
  struct timezone tz;
  double postEpochSecs;
  
  if (gettimeofday(&tv, &tz) < 0) {
    perror("GetTime");
    exit(-1);
  }
  
  postEpochSecs = (double)tv.tv_sec + ((double)tv.tv_usec/(double)1000000.0);
  return postEpochSecs;
}

double GetTimeMicroSeconds()
{
  struct timeval tv;
  struct timezone tz;
  double postEpochMicroSecs;
  
  if (gettimeofday(&tv, &tz) < 0) {
    perror("GetTimeMicroSeconds");
    exit(-1);
  }
  
  postEpochMicroSecs = (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
  return postEpochMicroSecs;

}

void PrintTimeStamp(struct timeval *ts)
{
  (void)printf("%02d:%02d:%02d.%06u ",
	       (unsigned int)ts->tv_sec / 3600,
	       ((unsigned int)ts->tv_sec % 3600) / 60,
	       (unsigned int)ts->tv_sec % 60, (unsigned int)ts->tv_usec);
}

void processBadPacket (struct IPPacket *p)
{

  if (session.debug == SESSION_DEBUG_HIGH) {
    printf("In ProcessBadPacket...\n");
  }
  /*
   * reset? the other guy does not like us?
   */
  if (INSESSION(p,session.dst,session.dport,session.src,session.sport) && (p->tcp->tcp_flags & TCPFLAGS_RST)) {
    printf("ERROR: EARLY_RST.\nRETURN CODE: %d\n", EARLY_RST);
    Quit(EARLY_RST);
  }
  /*
   * some other packet between us that is none of the above
   */
  if (INSESSION(p, session.src, session.sport, session.dst, session.dport) ||
      INSESSION(p, session.dst, session.dport, session.src, session.sport)) {

    printf("ERROR: Unexpected packet\nRETURN CODE: %d\n", UNEXPECTED_PKT);
    printf("Expecting:\n");
    printf("\tsrc = %s:%d (seq=%u, ack=%u)\n",
	   InetAddress(session.src), session.sport,
	   session.snd_nxt-session.iss,
	   session.rcv_nxt-session.irs);
    printf("\tdst = %s:%d (seq=%u, ack=%u)\n",
	   InetAddress(session.dst),session.dport,
	   session.rcv_nxt-session.irs, session.snd_una-session.iss);
    printf("Received:\n\t");
    PrintTcpPacket(p);
    printf ("session.snd_nxt=%d, session.rcv_nxt=%d, session.snd_una=%d\n", 
	    session.snd_nxt-session.iss, session.rcv_nxt-session.irs, session.snd_una-session.iss);
    Quit(UNEXPECTED_PKT);
  }
  /*
   * none of the above, 
   * so we must be seeing packets 
   * from some other flow!
   */
  else {
    printf("ERRROR: Received packet from different flow\nRETURN CODE: %d\n", DIFF_FLOW);
    PrintTcpPacket(p);
    Quit(DIFF_FLOW) ;
  }

  if (session.debug == SESSION_DEBUG_HIGH) {
    printf("Out ProcessBadPacket...\n");
  }
}

void busy_wait (double wait)
{
  double now = GetTime();
  double x = now ;
  while ((x - now) < wait) {
    x = GetTime();
  }
}
