/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <net/bpf.h>
#include <string.h>
#include <sys/socket.h>
#include <errno.h>
#include <net/if.h>
#include <stdbool.h>
#include "symbol_scope.h"

#include "bpflib.h"

#ifdef TESTING
#include "util.h"
#endif /* TESTING */

PRIVATE_EXTERN int
bpf_set_timeout(int fd, struct timeval * tv_p)
{
    return (ioctl(fd, BIOCSRTIMEOUT, tv_p));
}

PRIVATE_EXTERN int 
bpf_get_blen(int fd, int * blen)
{
    return(ioctl(fd, BIOCGBLEN, blen));
}

PRIVATE_EXTERN int
bpf_dispose(int bpf_fd)
{
    if (bpf_fd >= 0)
	return (close(bpf_fd));
    return (0);
}

PRIVATE_EXTERN int
bpf_new(void)
{
    char bpfdev[256];
    int i;
    int fd = -1;

    for (i = 0; true; i++) {
	snprintf(bpfdev, sizeof(bpfdev), "/dev/bpf%d", i);
	fd = open(bpfdev, O_RDWR , 0);
	if (fd >= 0) {
#ifdef SO_TC_CTL
	    int tc = SO_TC_CTL;
	    (void) ioctl(fd, BIOCSETTC, &tc);
#endif /* SO_TC_CTL */
	    break;
	}
	if (errno != EBUSY) {
	    break;
	}
    }
    return (fd);
}

PRIVATE_EXTERN int
bpf_setif(int fd, const char * en_name)
{
    struct ifreq ifr;

    strlcpy(ifr.ifr_name, en_name, sizeof(ifr.ifr_name));
    return (ioctl(fd, BIOCSETIF, &ifr));
}

PRIVATE_EXTERN int
bpf_set_immediate(int fd, u_int value)
{
    return (ioctl(fd, BIOCIMMEDIATE, &value));
}

PRIVATE_EXTERN int
bpf_set_hdrcmplt(int fd, u_int value)
{
    return (ioctl(fd, BIOCSHDRCMPLT, &value));
}

PRIVATE_EXTERN int
bpf_filter_receive_none(int fd)
{
    struct bpf_insn insns[] = {
	BPF_STMT(BPF_RET+BPF_K, 0),
    };
    struct bpf_program prog;

    prog.bf_len = sizeof(insns) / sizeof(struct bpf_insn);
    prog.bf_insns = insns;
    return ioctl(fd, BIOCSETF, &prog);
}

PRIVATE_EXTERN int
bpf_arp_filter(int fd, int type_offset, int type, int pkt_size)
{
    struct bpf_insn insns[] = {
	BPF_STMT(BPF_LD+BPF_H+BPF_ABS, type_offset),
	BPF_JUMP(BPF_JMP+BPF_JEQ+BPF_K, type, 0, 1),
	BPF_STMT(BPF_RET+BPF_K, pkt_size),
	BPF_STMT(BPF_RET+BPF_K, 0),
    };
    struct bpf_program prog;

    prog.bf_len = sizeof(insns) / sizeof(struct bpf_insn);
    prog.bf_insns = insns;
    return ioctl(fd, BIOCSETF, &prog);
}

PRIVATE_EXTERN int
bpf_write(int fd, void * pkt, int len)
{
    return ((int)write(fd, pkt, len));
}

#ifdef TESTING
#include <net/if_arp.h>
#include <net/ethernet.h>
#include <netinet/if_ether.h>


void
bpf_read_continuously(int fd, u_int blen)
{
    int n;
    char * rxbuf = malloc(blen);

    printf("rx buf len is %d\n", blen);
    while (1) {
	n = read(fd, rxbuf, blen);
	if (n < 0) {
	    perror("bpf_read_continuously");
	    return;
	}
	if (n == 0)
	    continue;
	print_data(rxbuf, n);
    }
}

int
main(int argc, char * argv[])
{
    int fd = bpf_new();
    char * en_name = "en0";
    u_int bpf_blen = 0;

    if (fd < 0) {
	perror("no bpf devices");
	exit(1);
    }
    
    if (argc > 1)
	en_name = argv[1];
    (void)bpf_set_immediate(fd, 1);
    if (bpf_arp_filter(fd, 12, ETHERTYPE_ARP, 
		       sizeof(struct ether_arp) + sizeof(struct ether_header)) 
	< 0) {
	perror("bpf_arp_filter");
    }
    if (bpf_setif(fd, en_name) < 0) {
	perror("bpf_attach");
	exit(1);
    }

    if (bpf_get_blen(fd, &bpf_blen) < 0) {
	perror("bpf_get_blen");
	exit(1);
    }
    bpf_read_continuously(fd, bpf_blen);
    exit(0);
    return (0);
}
#endif /* TESTING */
