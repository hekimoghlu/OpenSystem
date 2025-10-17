/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <termios.h>

#include "sudo_compat.h"

/* Non-standard termios input flags */
#ifndef IUCLC
# define IUCLC		0
#endif
#ifndef IMAXBEL
# define IMAXBEL	0
#endif

/* Non-standard termios local flags */
#ifndef IEXTEN
# define IEXTEN		0
#endif

/*
 * Set termios to raw mode (BSD extension).
 */
void
sudo_cfmakeraw(struct termios *term)
{
    /* Set terminal to raw mode */
    CLR(term->c_iflag,
	IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL|IXON|IMAXBEL|IUCLC);
    CLR(term->c_oflag, OPOST);
    CLR(term->c_lflag, ECHO|ECHONL|ICANON|ISIG|IEXTEN);
    CLR(term->c_cflag, CSIZE|PARENB);
    SET(term->c_cflag, CS8);
    term->c_cc[VMIN] = 1;
    term->c_cc[VTIME] = 0;
}
