/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#import "config.h"
#import "HangDetectionDisabler.h"

#if PLATFORM(MAC)

#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <wtf/ProcessPrivilege.h>
#import <wtf/RetainPtr.h>

namespace WebKit {

static const auto clientsMayIgnoreEventsKey = CFSTR("ClientMayIgnoreEvents");

static bool clientsMayIgnoreEvents()
{
    CFTypeRef valuePtr;
    if (CGSCopyConnectionProperty(CGSMainConnectionID(), CGSMainConnectionID(), clientsMayIgnoreEventsKey, &valuePtr) != kCGErrorSuccess)
        return false;

    return adoptCF(valuePtr) == kCFBooleanTrue;
}

static void setClientsMayIgnoreEvents(bool clientsMayIgnoreEvents)
{
    auto cgsId = CGSMainConnectionID();

    // In macOS 10.14 and later, the WebContent process does not have access to the WindowServer.
    // In this case, there will be no valid WindowServer main connection.
    if (!cgsId)
        return;
    // FIXME: <https://webkit.org/b/184484> We should assert here if this is being called from
    // the WebContent process.

    if (CGSSetConnectionProperty(cgsId, cgsId, clientsMayIgnoreEventsKey, clientsMayIgnoreEvents ? kCFBooleanTrue : kCFBooleanFalse) != kCGErrorSuccess)
        ASSERT_NOT_REACHED();
}

HangDetectionDisabler::HangDetectionDisabler()
    : m_clientsMayIgnoreEvents(clientsMayIgnoreEvents())
{
    setClientsMayIgnoreEvents(true);
}

HangDetectionDisabler::~HangDetectionDisabler()
{
    setClientsMayIgnoreEvents(m_clientsMayIgnoreEvents);
}

}

#endif
