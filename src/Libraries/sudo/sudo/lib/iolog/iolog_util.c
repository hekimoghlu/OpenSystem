/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <time.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_iolog.h"

/*
 * Map IOFD_* -> name.
 */
const char *
iolog_fd_to_name(int iofd)
{
    const char *ret;
    debug_decl(iolog_fd_to_name, SUDO_DEBUG_UTIL);

    switch (iofd) {
    case IOFD_STDIN:
	ret = "stdin";
	break;
    case IOFD_STDOUT:
	ret = "stdout";
	break;
    case IOFD_STDERR:
	ret = "stderr";
	break;
    case IOFD_TTYIN:
	ret = "ttyin";
	break;
    case IOFD_TTYOUT:
	ret = "ttyout";
	break;
    case IOFD_TIMING:
	ret = "timing";
	break;
    default:
	ret = "unknown";
	sudo_debug_printf(SUDO_DEBUG_ERROR, "%s: unexpected iofd %d",
	    __func__, iofd);
	break;
    }
    debug_return_const_str(ret);
}
