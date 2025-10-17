/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
/* Copyright (c) 1997 Apple Computer, Inc. All Rights Reserved */
/*
 * Copyright (c) 1988, Julian Onions <jpo@cs.nott.ac.uk>
 * Nottingham University 1987.
 *
 * This source may be freely distributed, however I would be interested
 * in any changes that are made.
 *
 * This driver takes packets off the IP i/f and hands them up to a
 * user process to have it's wicked way with. This driver has it's
 * roots in a similar driver written by Phil Cockcroft (formerly) at
 * UCL. This driver is based much more on read/write/select mode of
 * operation though.
 *
 */

#ifndef _NET_IF_TUN_H_
#define _NET_IF_TUN_H_
#include <sys/appleapiopts.h>

#ifdef KERNEL_PRIVATE
struct tun_softc {
	u_short tun_flags;              /* misc flags */
#define TUN_OPEN        0x0001
#define TUN_INITED      0x0002
#define TUN_RCOLL       0x0004
#define TUN_IASET       0x0008
#define TUN_DSTADDR     0x0010
#define TUN_RWAIT       0x0040
#define TUN_ASYNC       0x0080
#define TUN_NBIO        0x0100

#define TUN_READY       (TUN_OPEN | TUN_INITED | TUN_IASET)

	struct  ifnet tun_if;           /* the interface */
	int     tun_pgrp;               /* the process group - if any */
	struct  selinfo tun_rsel;       /* read select */
	struct  selinfo tun_wsel;       /* write select (not used) */
#if NBPFILTER > 0
	caddr_t         tun_bpf;
#endif
};

#endif /* KERNEL_PRIVATE */

/* ioctl's for get/set debug */
#define TUNSDEBUG       _IOW('t', 90, int)
#define TUNGDEBUG       _IOR('t', 89, int)

/* Maximum packet size */
#define TUNMTU          1500

#endif /* !_NET_IF_TUN_H_ */
