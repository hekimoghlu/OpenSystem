/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#include "SecItemShimProxy.h"

#if ENABLE(SEC_ITEM_SHIM)

#include "Connection.h"
#include "Logging.h"
#include "SecItemRequestData.h"
#include "SecItemResponseData.h"
#include "SecItemShimProxyMessages.h"
#include <Security/SecBase.h>
#include <Security/SecIdentity.h>
#include <Security/SecItem.h>
#include <WebCore/CertificateInfo.h>
#include <wtf/cf/VectorCF.h>

#if HAVE(SEC_KEYCHAIN)
#import <Security/SecKeychainItem.h>
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
WTF_DECLARE_CF_TYPE_TRAIT(SecKeychainItem);
ALLOW_DEPRECATED_DECLARATIONS_END
#endif

namespace WebKit {

#define MESSAGE_CHECK_COMPLETION(assertion, connection, completion) MESSAGE_CHECK_COMPLETION_BASE(assertion, connection, completion)

// We received these dictionaries over IPC so they shouldn't contain any "in-memory" objects (rdar://104253249).
static bool dictionaryContainsInMemoryObject(CFDictionaryRef dictionary)
{
    if (!dictionary)
        return false;

    // kSecUseItemList is deprecated on iOS 12+.
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if (CFDictionaryContainsKey(dictionary, kSecUseItemList))
        return true;
ALLOW_DEPRECATED_DECLARATIONS_END

    return CFDictionaryContainsKey(dictionary, kSecValueRef);
}

SecItemShimProxy& SecItemShimProxy::singleton()
{
    static SecItemShimProxy* proxy;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        proxy = new SecItemShimProxy;
    });
    return *proxy;
}

SecItemShimProxy::SecItemShimProxy()
    : m_queue(WorkQueue::create("com.apple.WebKit.SecItemShimProxy"_s))
{
}

SecItemShimProxy::~SecItemShimProxy()
{
    ASSERT_NOT_REACHED();
}

void SecItemShimProxy::initializeConnection(IPC::Connection& connection)
{
    connection.addMessageReceiver(m_queue.get(), *this, Messages::SecItemShimProxy::messageReceiverName());
}

void SecItemShimProxy::secItemRequest(IPC::Connection& connection, const SecItemRequestData& request, CompletionHandler<void(std::optional<SecItemResponseData>&&)>&& response)
{
    MESSAGE_CHECK_COMPLETION(!dictionaryContainsInMemoryObject(request.query()), connection, response(SecItemResponseData { errSecParam, nullptr }));
    MESSAGE_CHECK_COMPLETION(!dictionaryContainsInMemoryObject(request.attributesToMatch()), connection, response(SecItemResponseData { errSecParam, nullptr }));

    switch (request.type()) {
    case SecItemRequestData::Type::Invalid:
        LOG_ERROR("SecItemShimProxy::secItemRequest received an invalid data request. Please file a bug if you know how you caused this.");
        response(SecItemResponseData { errSecParam, nullptr });
        break;

    case SecItemRequestData::Type::CopyMatching: {
        CFTypeRef resultRawObject = nullptr;
        OSStatus resultCode = SecItemCopyMatching(request.query(), &resultRawObject);
        auto result = adoptCF(resultRawObject);

        SecItemResponseData::Result resultData;
        if (result) {
            auto resultType = CFGetTypeID(result.get());
            CFArrayRef resultArray = (CFArrayRef)result.get();
            if (resultType == CFArrayGetTypeID() && CFArrayGetCount(resultArray)) {
                auto containedType = CFGetTypeID(CFArrayGetValueAtIndex(resultArray, 0));
                if (containedType == SecCertificateGetTypeID()) {
                    resultData = Vector<RetainPtr<SecCertificateRef>>(makeVector(resultArray, [] (SecCertificateRef element) {
                        return std::optional(RetainPtr<SecCertificateRef> { element });
                    }));
#if HAVE(SEC_KEYCHAIN)
                    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
                } else if (containedType == SecKeychainItemGetTypeID()) {
                    resultData = Vector<RetainPtr<SecKeychainItemRef>>(makeVector(resultArray, [] (SecKeychainItemRef element) {
                        return std::optional(RetainPtr<SecKeychainItemRef> { element });
                    }));
                    ALLOW_DEPRECATED_DECLARATIONS_END
#endif
                } else
                    resultData = WTFMove(result);
            } else
                resultData = WTFMove(result);
        }
        response(SecItemResponseData { resultCode, WTFMove(resultData) });
        break;
    }

    case SecItemRequestData::Type::Add: {
        // Return value of SecItemAdd is often ignored. Even if it isn't, we don't have the ability to
        // serialize SecKeychainItemRef.
        OSStatus resultCode = SecItemAdd(request.query(), nullptr);
        response(SecItemResponseData { resultCode, nullptr });
        break;
    }

    case SecItemRequestData::Type::Update: {
        OSStatus resultCode = SecItemUpdate(request.query(), request.attributesToMatch());
        response(SecItemResponseData { resultCode, nullptr });
        break;
    }

    case SecItemRequestData::Type::Delete: {
        OSStatus resultCode = SecItemDelete(request.query());
        response(SecItemResponseData { resultCode, nullptr });
        break;
    }
    }
}

void SecItemShimProxy::secItemRequestSync(IPC::Connection& connection, const SecItemRequestData& data, CompletionHandler<void(std::optional<SecItemResponseData>&&)>&& completionHandler)
{
    secItemRequest(connection, data, WTFMove(completionHandler));
}

#undef MESSAGE_CHECK_COMPLETION

} // namespace WebKit

#endif // ENABLE(SEC_ITEM_SHIM)
