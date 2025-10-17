/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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

#include <sys/socket.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <pwd.h>
#include <unistd.h>
#if defined(HAVE_STDINT_H)
# include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
# include <inttypes.h>
#endif
#include <netinet/in.h>
#include <arpa/inet.h>
#ifdef NEED_RESOLV_H
# include <arpa/nameser.h>
# include <resolv.h>
#endif /* NEED_RESOLV_H */
#include <netdb.h>

#include "sudoers.h"
#include "interfaces.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

struct interface_list *
get_interfaces(void)
{
    static struct interface_list empty = SLIST_HEAD_INITIALIZER(interfaces);
    return &empty;
}

void
init_eventlog_config(void)
{
    return;
}

int
group_plugin_query(const char *user, const char *group, const struct passwd *pw)
{
    return false;
}

bool
set_perms(int perm)
{
    return true;
}

bool
restore_perms(void)
{
    return true;
}

bool
rewind_perms(void)
{
    return true;
}

bool
sudo_nss_can_continue(struct sudo_nss *nss, int match)
{
    return true;
}
