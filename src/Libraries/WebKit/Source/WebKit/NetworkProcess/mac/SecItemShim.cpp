/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#include "SecItemShim.h"

#if ENABLE(SEC_ITEM_SHIM)

#include "BlockingResponseMap.h"
#include "NetworkProcess.h"
#include "SecItemRequestData.h"
#include "SecItemResponseData.h"
#include "SecItemShimProxyMessages.h"
#include <Security/Security.h>
#include <atomic>
#include <dlfcn.h>
#include <mutex>
#include <wtf/ProcessPrivilege.h>
#include <wtf/cf/VectorCF.h>
#include <wtf/threads/BinarySemaphore.h>

#if USE(APPLE_INTERNAL_SDK)
#include <CFNetwork/CFURLConnectionPriv.h>
#else
struct _CFNFrameworksStubs {
    CFIndex version;

    OSStatus (*SecItem_stub_CopyMatching)(CFDictionaryRef query, CFTypeRef *result);
    OSStatus (*SecItem_stub_Add)(CFDictionaryRef attributes, CFTypeRef *result);
    OSStatus (*SecItem_stub_Update)(CFDictionaryRef query, CFDictionaryRef attributesToUpdate);
    OSStatus (*SecItem_stub_Delete)(CFDictionaryRef query);
};
#endif

extern "C" void _CFURLConnectionSetFrameworkStubs(const struct _CFNFrameworksStubs* stubs);

namespace WebKit {

static WeakPtr<NetworkProcess>& globalNetworkProcess()
{
    static NeverDestroyed<WeakPtr<NetworkProcess>> networkProcess;
    return networkProcess.get();
}

static std::optional<SecItemResponseData> sendSecItemRequest(SecItemRequestData::Type requestType, CFDictionaryRef query, CFDictionaryRef attributesToMatch = 0)
{
    if (RunLoop::isMain()) {
        auto sendSync = globalNetworkProcess()->parentProcessConnection()->sendSync(Messages::SecItemShimProxy::SecItemRequestSync(SecItemRequestData(requestType, query, attributesToMatch)), 0);
        auto [response] = sendSync.takeReplyOr(std::nullopt);
        return response;
    }

    std::optional<SecItemResponseData> response;
    BinarySemaphore semaphore;

    RunLoop::main().dispatch([&] {
        if (!globalNetworkProcess()) {
            semaphore.signal();
            return;
        }

        globalNetworkProcess()->parentProcessConnection()->sendWithAsyncReply(Messages::SecItemShimProxy::SecItemRequest(SecItemRequestData(requestType, query, attributesToMatch)), [&](auto reply) {
            if (reply)
                response = WTFMove(*reply);

            semaphore.signal();
        });
    });

    semaphore.wait();

    return response;
}

static OSStatus webSecItemCopyMatching(CFDictionaryRef query, CFTypeRef* result)
{
    auto response = sendSecItemRequest(SecItemRequestData::Type::CopyMatching, query);
    if (!response)
        return errSecInteractionNotAllowed;

    WTF::switchOn(response->resultObject(), [&] (std::nullptr_t) {
        *result = nullptr;
    }, [&] (Vector<RetainPtr<SecCertificateRef>>& certificates) {
        *result = createCFArray(certificates, [] (auto& certificate) {
            return certificate.get();
        }).leakRef();
#if HAVE(SEC_KEYCHAIN)
    }, [&] (Vector<RetainPtr<SecKeychainItemRef>>& items) {
        *result = createCFArray(items, [] (auto& item) {
            return item.get();
        }).leakRef();
#endif
    }, [&] (RetainPtr<CFTypeRef>& type) {
        *result = type.leakRef();
    });

    return response->resultCode();
}

static OSStatus webSecItemAdd(CFDictionaryRef query, CFTypeRef* unusedResult)
{
    // Return value of SecItemAdd should be ignored for WebKit use cases. WebKit can't serialize SecKeychainItemRef, so we do not use it.
    // If someone passes a result value to be populated, the API contract is being violated so we should assert.
    if (unusedResult) {
        ASSERT_NOT_REACHED();
        return errSecParam;
    }

    auto response = sendSecItemRequest(SecItemRequestData::Type::Add, query);
    if (!response)
        return errSecInteractionNotAllowed;

    return response->resultCode();
}

static OSStatus webSecItemUpdate(CFDictionaryRef query, CFDictionaryRef attributesToUpdate)
{
    auto response = sendSecItemRequest(SecItemRequestData::Type::Update, query, attributesToUpdate);
    if (!response)
        return errSecInteractionNotAllowed;
    
    return response->resultCode();
}

static OSStatus webSecItemDelete(CFDictionaryRef query)
{
    auto response = sendSecItemRequest(SecItemRequestData::Type::Delete, query);
    if (!response)
        return errSecInteractionNotAllowed;
    
    return response->resultCode();
}

void initializeSecItemShim(NetworkProcess& process)
{
    globalNetworkProcess() = process;

    struct _CFNFrameworksStubs stubs = {
        .version = 0,
        .SecItem_stub_CopyMatching = webSecItemCopyMatching,
        .SecItem_stub_Add = webSecItemAdd,
        .SecItem_stub_Update = webSecItemUpdate,
        .SecItem_stub_Delete = webSecItemDelete,
    };

    _CFURLConnectionSetFrameworkStubs(&stubs);
}

} // namespace WebKit

#endif // ENABLE(SEC_ITEM_SHIM)
