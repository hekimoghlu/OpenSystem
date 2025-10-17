/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#include "login_locl.h"

RCSID("$Id$");

#include <termios.h>
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
 */#ifndef FLUSHO
#define	FLUSHO	0
#endif

#ifndef XTABS
#define	XTABS	0
#endif

#ifndef OXTABS
#define OXTABS	XTABS
#endif

/* Ultrix... */
#ifndef ECHOPRT
#define ECHOPRT	0
#endif

#ifndef ECHOCTL
#define ECHOCTL	0
#endif

#ifndef ECHOKE
#define ECHOKE	0
#endif

#ifndef IMAXBEL
#define IMAXBEL	0
#endif

#define Ctl(x) ((x) ^ 0100)

void
stty_default(void)
{
    struct	termios termios;

    /*
     * Finalize the terminal settings. Some systems default to 8 bits,
     * others to 7, so we should leave that alone.
     */
    tcgetattr(0, &termios);

    termios.c_iflag |= (BRKINT|IGNPAR|ICRNL|IXON|IMAXBEL);
    termios.c_iflag &= ~IXANY;

    termios.c_lflag |= (ISIG|IEXTEN|ICANON|ECHO|ECHOE|ECHOK|ECHOCTL|ECHOKE);
    termios.c_lflag &= ~(ECHOPRT|TOSTOP|FLUSHO);

    termios.c_oflag |= (OPOST|ONLCR);
    termios.c_oflag &= ~OXTABS;

    termios.c_cc[VINTR] = Ctl('C');
    termios.c_cc[VERASE] = Ctl('H');
    termios.c_cc[VKILL] = Ctl('U');
    termios.c_cc[VEOF] = Ctl('D');

    termios.c_cc[VSUSP] = Ctl('Z');

    tcsetattr(0, TCSANOW, &termios);
}

