/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
//
// auditevents - monitor and act upon audit subsystem events
//
#include <os/log.h>
#include "auditevents.h"
#include "dtrace.h"
#include <security_utilities/logging.h>
#include "self.h"

using namespace UnixPlusPlus;
using namespace MachPlusPlus;


AuditMonitor::AuditMonitor(Port relay)
	: Thread("AuditMonitor"), mRelay(relay)
{
}

AuditMonitor::~AuditMonitor()
{
}


//
// Endlessly retrieve session events and dispatch them.
// (The current version of MachServer cannot receive FileDesc-based events,
// so we need a monitor thread for this.)
//
void AuditMonitor::threadAction()
{
    au_sdev_handle_t *dev;
	int event;
	auditinfo_addr_t aia;

    // This retries forever since securityd can't functions correctly without getting audit sessions events
    while (1) {
        dev = au_sdev_open(AU_SDEVF_ALLSESSIONS);
        if (NULL == dev) {
            os_log_fault(OS_LOG_DEFAULT, "auditevents count not open audit device: %d, retrying in a bit", errno);
            sleep(10);
            continue;
        }

        for (;;) {
            if (0 != au_sdev_read_aia(dev, &event, &aia)) {
                secerror("au_sdev_read_aia failed: %d\n", errno);
                break;
            }
            secinfo("SecServer", "%p session notify %d %d %d", this, aia.ai_asid, event, aia.ai_auid);
            if (kern_return_t rc = self_client_handleSession(mRelay, event, aia.ai_asid)) {
                secerror("self-send failed (mach error %d)", rc);
            }
        }
        au_sdev_close(dev);
    }
}
