/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
#include <sys/vnode.h>
#include <sys/kauth.h>
#include <sys/param.h>
#include <sys/tty.h>
#include <security/mac_framework.h>
#include <security/mac_internal.h>

void
mac_pty_notify_grant(proc_t p, struct tty *tp, dev_t dev, struct label *label)
{
	MAC_PERFORM(pty_notify_grant, p, tp, dev, label);
}

void
mac_pty_notify_close(proc_t p, struct tty *tp, dev_t dev, struct label *label)
{
	MAC_PERFORM(pty_notify_close, p, tp, dev, label);
}
