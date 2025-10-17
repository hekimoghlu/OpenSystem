/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
#include "NetworkStateNotifier.h"

#include <wtf/MainThread.h>
#include <wtf/Vector.h>

#include <winsock2.h>
#include <iphlpapi.h>

namespace WebCore {

void NetworkStateNotifier::updateStateWithoutNotifying()
{
    DWORD size = 0;
    if (::GetAdaptersAddresses(AF_UNSPEC, 0, 0, 0, &size) != ERROR_BUFFER_OVERFLOW)
        return;

    Vector<char> buffer(size);
    auto addresses = reinterpret_cast<PIP_ADAPTER_ADDRESSES>(buffer.data());
    if (::GetAdaptersAddresses(AF_UNSPEC, 0, 0, addresses, &size) != ERROR_SUCCESS)
        return;

    for (; addresses; addresses = addresses->Next) {
        if (addresses->IfType != MIB_IF_TYPE_LOOPBACK && addresses->OperStatus == IfOperStatusUp) {
            // We found an interface that was up.
            m_isOnLine = true;
            return;
        }
    }

    m_isOnLine = false;
}

void CALLBACK NetworkStateNotifier::addressChangeCallback(void*, BOOLEAN)
{
    callOnMainThread([] {
        // NotifyAddrChange only notifies us of a single address change. Now that we've been notified,
        // we need to call it again so we'll get notified the *next* time.
        singleton().registerForAddressChange();

        singleton().updateStateSoon();
    });
}

void NetworkStateNotifier::registerForAddressChange()
{
    HANDLE handle;
    ::NotifyAddrChange(&handle, &m_overlapped);
}

void NetworkStateNotifier::startObserving()
{
    memset(&m_overlapped, 0, sizeof(m_overlapped));
    m_overlapped.hEvent = ::CreateEvent(0, false, false, 0);
    ::RegisterWaitForSingleObject(&m_waitHandle, m_overlapped.hEvent, addressChangeCallback, nullptr, INFINITE, 0);
    registerForAddressChange();
}

} // namespace WebCore
