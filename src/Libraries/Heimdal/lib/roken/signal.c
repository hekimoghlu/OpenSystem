/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#include <config.h>

#include <signal.h>
#include "roken.h"

/*
 * We would like to always use this signal but there is a link error
 * on NEXTSTEP
 */
#if !defined(NeXT) && !defined(__APPLE__)
/*
 * Bugs:
 *
 * Do we need any extra hacks for SIGCLD and/or SIGCHLD?
 */

ROKEN_LIB_FUNCTION SigAction ROKEN_LIB_CALL
signal(int iSig, SigAction pAction)
{
    struct sigaction saNew, saOld;

    saNew.sa_handler = pAction;
    sigemptyset(&saNew.sa_mask);
    saNew.sa_flags = 0;

    if (iSig == SIGALRM)
	{
#ifdef SA_INTERRUPT
	    saNew.sa_flags |= SA_INTERRUPT;
#endif
	}
    else
	{
#ifdef SA_RESTART
	    saNew.sa_flags |= SA_RESTART;
#endif
	}

    if (sigaction(iSig, &saNew, &saOld) < 0)
	return(SIG_ERR);

    return(saOld.sa_handler);
}
#endif
