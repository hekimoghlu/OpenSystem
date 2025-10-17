/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
/*
 * Copyright (c) 1987 NeXT, Inc.
 */

#include <sys/_types/_dev_t.h>     /* dev_t */

struct consdev {
	char    *cn_name;       /* name of device in dev_name_list */
	int     (*cn_probe)(void);      /* probe and fill in consdev info */
	int     (*cn_init)(void);       /* turn on as console */
	int     (*cn_getc)(void);       /* kernel getchar interface */
	int     (*cn_putc)(void);       /* kernel putchar interface */
	struct  tty *cn_tp;     /* tty structure for console device */
	dev_t   cn_dev;         /* major/minor of device */
	short   cn_pri;         /* pecking order; the higher the better */
};

/* values for cn_pri - reflect our policy for console selection */
#define CN_DEAD         0       /* device doesn't exist */
#define CN_NORMAL       1       /* device exists but is nothing special */
#define CN_INTERNAL     2       /* "internal" bit-mapped display */
#define CN_REMOTE       3       /* serial interface with remote bit set */

/* XXX */
#define CONSMAJOR       0

#ifdef KERNEL

#include <sys/types.h>
#include <sys/conf.h>

extern  struct consdev constab[];
extern  struct consdev *cn_tab;
extern  struct tty *cn_tty;

/* Returns tty with +1 retain count, use ttyfree() to release. */
extern struct tty       *copy_constty(void);               /* current console device */
extern struct tty       *set_constty(struct tty *);

int consopen(dev_t, int, int, struct proc *);
int consclose(dev_t, int, int, struct proc *);
int consread(dev_t, struct uio *, int);
int conswrite(dev_t, struct uio *, int);
int consioctl(dev_t, u_long, caddr_t, int, struct proc *);
int consselect(dev_t, int, void *, struct proc *);

/*
 * These really want their own header file, but this is the only one in
 * common, and the km device is the keyboard monitor, so it's technically a
 * part of the console.
 */
int kmopen(dev_t, int, int, struct proc *);
int kmclose(dev_t, int, int, struct proc *);
int kmread(dev_t, struct uio *, int);
int kmwrite(dev_t, struct uio *, int);
int kmioctl(dev_t, u_long, caddr_t, int, struct proc *);
int kmputc(dev_t, char);

#endif
