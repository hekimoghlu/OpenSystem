/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#include <sys/sysctl.h>
#include <net/bpf.h>
#include <net/if.h>
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>

#include "netstat.h"

static void
proc_name(pid_t pid, char *buf, size_t buf_len)
{
	int name[4];
	size_t	namelen, infolen;
	struct kinfo_proc info;

	name[0] = CTL_KERN;
	name[1] = KERN_PROC;
	name[2] = KERN_PROC_PID;
	name[3] = pid;
	namelen = 4;
	infolen = sizeof(info);
	if (sysctl(name, namelen, &info, &infolen, 0, 0) != 0) {
		snprintf(buf, buf_len, "");
		return;
	}
	snprintf(buf, buf_len, "%s", info.kp_proc.p_comm);
}

static void
bd_flags(struct xbpf_d *bd, char *flagbuf, size_t len)
{
	snprintf(flagbuf, len, "%c%c%c%c%c%c%c%c%c%c%c%c",
		 bd->bd_promisc ? 'p' : '-',
		 bd->bd_immediate ? 'i' : '-',
		 bd->bd_hdrcmplt ? '-' : 'f',
		 bd->bd_async ? 'a' : '-',
		 (bd->bd_direction & BPF_D_IN) ? 'I' : '-',
		 (bd->bd_direction & BPF_D_OUT) ? 'O' : '-',
		 bd->bd_headdrop ? 'h' : '-',
		 bd->bh_compreq ? (bd->bh_compenabled ? 'C' : 'c') : '-',
		 bd->bd_exthdr ? 'x' : '-',
		 bd->bd_trunc ? 't' : '-',
		 bd->bd_pkthdrv2 ? '2' : '-',
#ifdef BIOCGDVRTIN
		 bd->bd_divert_in ? 'D' : '-'
#else /* BIOCGDVRTIN */
		 '-'
#endif /* BIOCGDVRTIN */
		 );
}

void
bpf_stats(char *interface)
{
	size_t len;
	void *buffer;
	struct xbpf_d *bd;

	if (sysctlbyname("debug.bpf_stats", NULL, &len, NULL, 0) != 0) {
		err(EX_OSERR, "sysctlbyname debug.bpf_stats");
	}
	if (len == 0) {
		return;
	}
	buffer = malloc(2 * len);
	if (buffer == NULL) {
		err(EX_OSERR, "malloc");
	}
	if (sysctlbyname("debug.bpf_stats", buffer, &len, NULL, 0) != 0) {
		err(EX_OSERR, "sysctlbyname debug.bpf_stats");
	}
	printf("%-9s %-14s %-13s %9s %9s %9s %12s %9s %9s %9s %9s %9s %9s %9s %12s %9s %9s %s\n",
	       "Device", "Netif", "Flags",
	       "Recv", "RDrop", "RMatch", "RSize",
	       "ReadCnt",
	       "Bsize", "Sblen", "Scnt", "Hblen", "Hcnt",
	       "Ccnt", "Csize",
	       "Written", "WDrop",
	       "Command");
	for (bd = (struct xbpf_d *)buffer;
	     (void *)(bd + 1) <= buffer + len;
	     bd = (void *)bd + bd->bd_structsize) {
		char flagbuf[32];
		char namebuf[32];

		if (interface != NULL &&
		    strncmp(interface, bd->bd_ifname, sizeof(bd->bd_ifname)) != 0) {
			continue;
		}

		bd_flags(bd, flagbuf, sizeof(flagbuf));
		proc_name(bd->bd_pid, namebuf, sizeof(namebuf));
		printf("bpf%-6u %-14s %11s %9llu %9llu %9llu %12llu %9llu %9u %9u %9u %9u %9u %9llu %12llu %9llu %9llu %s.%d\n",
		       bd->bd_dev_minor, bd->bd_ifname, flagbuf,
		       bd->bd_rcount, bd->bd_dcount, bd->bd_fcount, bd->bd_fsize,
		       bd->bd_read_count,
		       bd->bd_bufsize, bd->bd_slen, bd->bd_scnt, bd->bd_hlen, bd->bd_hcnt,
		       bd->bd_comp_count, bd->bd_comp_size,
		       bd->bd_wcount, bd->bd_wdcount,
		       namebuf, bd->bd_pid);
	}

	free(buffer);
}

void
bpf_help(void)
{
	printf("Meaning of the letters in the Flags column of the statistics about BPF devices:\n");

#define FMT "\t%c\t%s\n"

	printf(FMT, 'p', "promiscuous mode enabled");
	printf(FMT, 'i', "immediate mode enabled");
	printf(FMT, 'h', "header complete enabled");
	printf(FMT, 'a', "asynchronous mode enabled");
	printf(FMT, 'I', "see incoming packets");
	printf(FMT, 'O', "see outgoing packets");
	printf(FMT, 'h', "head drop mode");
	printf(FMT, 'C', "compression enabled");
	printf(FMT, 'c', "compression requested");
	printf(FMT, 'x', "use BPF extended header");
	printf(FMT, 't', "truncation requested");
	printf(FMT, '2', "pktap header v2");
	printf(FMT, 'B', "batch write enabled");
#ifdef BIOCGDVRTIN
	printf(FMT, 'D', "divert input mode");
#endif /* BIOCGDVRTIN */
	printf(FMT, '-', "option not enabled");

#undef FMT
}
