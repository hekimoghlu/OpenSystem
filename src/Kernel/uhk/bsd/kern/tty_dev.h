/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#ifndef __TTY_DEV_H__
#define __TTY_DEV_H__

/*
 * ptmx_ioctl is a pointer to a list of pointers to tty structures which is
 * grown, as necessary, copied, and replaced, but never shrunk.  The ioctl
 * structures themselves pointed to from this list come and go as needed.
 */
struct ptmx_ioctl {
	struct tty      *pt_tty;        /* pointer to ttymalloc()'ed data */
	int             pt_flags;
	struct selinfo  pt_selr;
	struct selinfo  pt_selw;
	u_char          pt_send;
	u_char          pt_ucntl;
	void            *pt_devhandle;  /* cloned replica device handle */
};

#define PF_PKT          0x0008          /* packet mode */
#define PF_STOPPED      0x0010          /* user told stopped */
#define PF_NOSTOP       0x0040
#define PF_UCNTL        0x0080          /* user control mode */
#define PF_UNLOCKED     0x0100          /* replica unlock (primary open resets) */
#define PF_OPEN_M       0x0200          /* primary is open */
#define PF_OPEN_S       0x0400          /* replica is open */

struct tty_dev_t {
	int primary;     // primary major device number
	int replica;     // replica major device number
	unsigned int    fix_7828447:1,
	    fix_7070978:1,
	    mac_notify:1,
	    open_reset:1,
	    _reserved:28;
#if __LP64__
	int _pad;
#endif

	struct tty_dev_t *next;

	struct ptmx_ioctl *(*open)(int minor, int flags);
	int (*free)(int minor, int flags);
	int (*name)(int minor, char *buffer, size_t size);
	void (*revoke)(int minor, struct tty *tp);
};

extern void tty_dev_register(struct tty_dev_t *dev);

extern int ttnread(struct tty *tp);

extern void termios32to64(struct termios32 *in, struct user_termios *out);
extern void termios64to32(struct user_termios *in, struct termios32 *out);

#endif // __TTY_DEV_H__
