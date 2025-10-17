/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#include <stdlib.h>
#include <stdio.h>
#include "gmt2local.h"
#include "pcap.h"
#include "inet.h"
#include "capture.h"

/* set snaplen to max etherenet packet size */
#define DEFAULT_SNAPLEN 1500 

pcap_t *pc;		/* pcap device */
int datalinkOffset;	/* offset of ip packet from datalink packet */
int captureDebug = 1;
unsigned int thisTimeZone;

void CaptureInit(u_int32_t sourceIP, u_int16_t sourcePort,
		 u_int32_t targetIP, u_int16_t targetPort, char *dev)
{

  char *device = NULL;
  char errbuf[PCAP_ERRBUF_SIZE];
  int snaplen = DEFAULT_SNAPLEN;
  int promisc = 1;
  int timeout = 10;  /* timeout in 1 second (10 ms) */
  char filtercmds[255];
  bpf_u_int32 netmask = 0;
  struct bpf_program filter;
  char source[18];
  char target[18];
  int i;

  /* Get local time zone for interpreting timestamps */
  /* XXX - does this belong here? */
  thisTimeZone = gmt2local(0);

  if (dev != NULL) {
    device = dev;
  } else {
    pcap_if_t *devlist;
    /*
    * Find the list of interfaces, and pick
     * the first interface.
     */
    if (pcap_findalldevs(&devlist, errbuf) >= 0 &&
	      devlist != NULL) {
      device = strdup(devlist->name);
      pcap_freealldevs(devlist);
    }

    if (device == NULL) {
      fprintf(stderr, "Can't find capture device: %s\n", errbuf);
      exit(-1);
    }
  }
 
  if (captureDebug) {
    printf("Device name is %s\n", device);
  }
  pc = pcap_open_live(device, snaplen, promisc, timeout, errbuf);
  if (pc == NULL) {
    fprintf(stderr,"Can't open capture device %s: %s\n",device, errbuf);
    exit(-1);
  } 

  /* XXX why do we need to do this? */
  i = pcap_snapshot(pc);
  if (snaplen < i) {
    fprintf(stderr, "Warning: snaplen raised to %d from %d",
	    snaplen, i);
  }

  if ((i = pcap_datalink(pc)) < 0) {
    fprintf(stderr,"Unable to determine datalink type for %s: %s\n",
	    device, errbuf);
    exit(-1);
  }

  switch(i) {

    case DLT_EN10MB: datalinkOffset = 14; break;
    case DLT_IEEE802: datalinkOffset = 22; break;
    case DLT_NULL: datalinkOffset = 4; break;
    case DLT_SLIP: 
    case DLT_PPP: datalinkOffset = 24; break;
    case DLT_RAW: datalinkOffset = 0; break;
    default: 
       fprintf(stderr,"Unknown datalink type %d\n",i);
       exit(-1);
       break;

  }

  if (InetAddress(sourceIP) < 0) {
    fprintf(stderr, "Invalid source IP address (%d)\n", sourceIP);
    exit(-1);
  }

  strlcpy(source, InetAddress(sourceIP), sizeof(source));
  strlcpy(target, InetAddress(targetIP), sizeof(target));

  /* Setup initial filter */
  snprintf(filtercmds, sizeof(filtercmds),
    "(host %s && host %s && port %d) || icmp\n",
    source, target, targetPort);

  if (captureDebug) {
    printf("datalinkOffset = %d\n", datalinkOffset);
    printf("filter = %s\n", filtercmds);
  }
  if (pcap_compile(pc, &filter, filtercmds, 1, netmask) < 0) {
    printf("Error: %s", pcap_geterr(pc));
    exit(-1);
  }

  if (pcap_setfilter(pc, &filter) < 0) {
    fprintf(stderr, "Can't set filter: %s",pcap_geterr(pc));
    exit(-1);
  }
  
  if (captureDebug) {
    printf("Listening on %s...\n", device);
  }

}

char *CaptureGetPacket(struct pcap_pkthdr *pi)
{

  const u_char *p;

  p = pcap_next(pc, (struct pcap_pkthdr *)pi);

  if (p != NULL) {
    p += datalinkOffset;
  }

  pi->ts.tv_sec = (pi->ts.tv_sec + thisTimeZone) % 86400;

  return (char *)p;

}


void CaptureEnd()
{
  struct pcap_stat stat;

  if (pcap_stats(pc, &stat) < 0) {
    (void)fprintf(stderr, "pcap_stats: %s\n", pcap_geterr(pc));
  }
  else {
    (void)fprintf(stderr, "%d packets received by filter\n", stat.ps_recv); 
    (void)fprintf(stderr, "%d packets dropped by kernel\n", stat.ps_drop);
  }

  pcap_close(pc);
}

