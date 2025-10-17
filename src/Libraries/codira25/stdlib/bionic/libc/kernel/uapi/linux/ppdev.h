/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#ifndef _UAPI_LINUX_PPDEV_H
#define _UAPI_LINUX_PPDEV_H
#define PP_IOCTL 'p'
#define PPSETMODE _IOW(PP_IOCTL, 0x80, int)
#define PPRSTATUS _IOR(PP_IOCTL, 0x81, unsigned char)
#define PPWSTATUS OBSOLETE__IOW(PP_IOCTL, 0x82, unsigned char)
#define PPRCONTROL _IOR(PP_IOCTL, 0x83, unsigned char)
#define PPWCONTROL _IOW(PP_IOCTL, 0x84, unsigned char)
struct ppdev_frob_struct {
  unsigned char mask;
  unsigned char val;
};
#define PPFCONTROL _IOW(PP_IOCTL, 0x8e, struct ppdev_frob_struct)
#define PPRDATA _IOR(PP_IOCTL, 0x85, unsigned char)
#define PPWDATA _IOW(PP_IOCTL, 0x86, unsigned char)
#define PPRECONTROL OBSOLETE__IOR(PP_IOCTL, 0x87, unsigned char)
#define PPWECONTROL OBSOLETE__IOW(PP_IOCTL, 0x88, unsigned char)
#define PPRFIFO OBSOLETE__IOR(PP_IOCTL, 0x89, unsigned char)
#define PPWFIFO OBSOLETE__IOW(PP_IOCTL, 0x8a, unsigned char)
#define PPCLAIM _IO(PP_IOCTL, 0x8b)
#define PPRELEASE _IO(PP_IOCTL, 0x8c)
#define PPYIELD _IO(PP_IOCTL, 0x8d)
#define PPEXCL _IO(PP_IOCTL, 0x8f)
#define PPDATADIR _IOW(PP_IOCTL, 0x90, int)
#define PPNEGOT _IOW(PP_IOCTL, 0x91, int)
#define PPWCTLONIRQ _IOW(PP_IOCTL, 0x92, unsigned char)
#define PPCLRIRQ _IOR(PP_IOCTL, 0x93, int)
#define PPSETPHASE _IOW(PP_IOCTL, 0x94, int)
#define PPGETTIME _IOR(PP_IOCTL, 0x95, struct timeval)
#define PPSETTIME _IOW(PP_IOCTL, 0x96, struct timeval)
#define PPGETMODES _IOR(PP_IOCTL, 0x97, unsigned int)
#define PPGETMODE _IOR(PP_IOCTL, 0x98, int)
#define PPGETPHASE _IOR(PP_IOCTL, 0x99, int)
#define PPGETFLAGS _IOR(PP_IOCTL, 0x9a, int)
#define PPSETFLAGS _IOW(PP_IOCTL, 0x9b, int)
#define PP_FASTWRITE (1 << 2)
#define PP_FASTREAD (1 << 3)
#define PP_W91284PIC (1 << 4)
#define PP_FLAGMASK (PP_FASTWRITE | PP_FASTREAD | PP_W91284PIC)
#endif
