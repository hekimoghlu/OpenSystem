/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#ifndef _SGTTY_H_
#define _SGTTY_H_

#include <sys/cdefs.h>

#ifndef USE_OLD_TTY
#define	USE_OLD_TTY
#endif
#include <sys/ioctl.h>

__BEGIN_DECLS
int	gtty(int, struct sgttyb *);
int	stty(int, struct sgttyb *);
__END_DECLS

#define	gtty(fd, buf)	ioctl(fd, TIOCGETP, buf)
#define	stty(fd, buf)	ioctl(fd, TIOCSETP, buf)

#endif /* _SGTTY_H_ */
