/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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

#include "includes.h"

#ifdef __APPLE_CLEAR_LV__
#include <dlfcn.h>
#include <unistd.h>
#include <sys/fcntl.h>
#include <mach-o/dyld_priv.h>
#include <System/sys/codesign.h>
#include "log.h"

/*
 * <rdar://problem/65693657> [sshd] Adopt
 * com.apple.private.security.clear-library-validation. Attempt to
 * dynamically load a module. Disable LV on the process if necessary.
 * NB: Code is based on OpenPAM's openpam_dlopen().
 */

void *
dlopen_lv(char *path, int mode)
{
    /* Fast path: dyld shared cache. */
    if (_dyld_shared_cache_contains_path(path)) {
        return dlopen(path, mode);
    }

    /* Slow path: check file on disk. */
    if (faccessat(AT_FDCWD, path, R_OK, AT_EACCESS) != 0) {
        return NULL;
    }

    void *dlh = dlopen(path, mode);
    if (dlh != NULL) {
        return dlh;
    }

    /*
     * The module exists and is readable, but failed to load.  If
     * library validation is enabled, try disabling it and then try
     * again.
     */
    int   csflags = 0;
    pid_t pid     = getpid();
    csops(pid, CS_OPS_STATUS, &csflags, sizeof(csflags));
    if ((csflags & (CS_FORCED_LV | CS_REQUIRE_LV)) == 0) {
        return NULL;
    }

    int rv = csops(getpid(), CS_OPS_CLEAR_LV, NULL, 0);
    if (rv != 0) {
        error("csops(CS_OPS_CLEAR_LV) failed: %d", rv);
        return NULL;
    }

    dlh = dlopen(path, mode);
    if (dlh == NULL) {
        /* Failed to load even with LV disabled: re-enable LV. */
        csflags = CS_REQUIRE_LV;
        csops(pid, CS_OPS_SET_STATUS, &csflags, sizeof(csflags));
    }

    return dlh;
}
#endif /* __APPLE_CLEAR_LV__ */
