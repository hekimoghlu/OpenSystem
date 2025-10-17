/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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
#include "config.h"
#include "PowerSourceNotifier.h"

#import "SystemBattery.h"
#import <notify.h>
#import <pal/spi/cocoa/IOPSLibSPI.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PowerSourceNotifier);

PowerSourceNotifier::PowerSourceNotifier(PowerSourceNotifierCallback&& callback)
    : m_callback(WTFMove(callback))
{
    int token = 0;
    auto status = notify_register_dispatch(kIOPSNotifyPowerSource, &token, dispatch_get_main_queue(), [weakThis = WeakPtr { *this }] (int) {
        if (weakThis)
            weakThis->notifyPowerSourceChanged();
    });
    if (status == NOTIFY_STATUS_OK)
        m_tokenID = token;

    // If the current value of systemHasAC() is uncached, force a notification.
    if (!cachedSystemHasAC()) {
        RunLoop::main().dispatch([weakThis = WeakPtr { *this }] {
            if (weakThis)
                weakThis->notifyPowerSourceChanged();
        });
    }
}

PowerSourceNotifier::~PowerSourceNotifier()
{
    if (m_tokenID)
        notify_cancel(*m_tokenID);
}

void PowerSourceNotifier::notifyPowerSourceChanged()
{
    resetSystemHasAC();
    if (m_callback)
        m_callback(systemHasAC());
}

}
