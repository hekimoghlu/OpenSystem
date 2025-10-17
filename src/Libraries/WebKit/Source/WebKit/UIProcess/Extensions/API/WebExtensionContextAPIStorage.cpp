/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
#include "WebExtensionContext.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include "WebExtensionConstants.h"
#include "WebExtensionContextProxy.h"
#include "WebExtensionContextProxyMessages.h"
#include "WebExtensionDataType.h"
#include "WebExtensionPermission.h"
#include "WebExtensionStorageAccessLevel.h"
#include "WebExtensionStorageSQLiteStore.h"
#include "WebExtensionUtilities.h"
#include <wtf/text/MakeString.h>

namespace WebKit {

bool WebExtensionContext::isStorageMessageAllowed()
{
    return isLoaded() && (hasPermission(WebExtensionPermission::storage()) || hasPermission(WebExtensionPermission::unlimitedStorage()));
}

void WebExtensionContext::storageGet(WebPageProxyIdentifier webPageProxyIdentifier, WebExtensionDataType dataType, const Vector<String>& keys, CompletionHandler<void(Expected<String, WebExtensionError>&&)>&& completionHandler)
{
    auto callingAPIName = makeString("browser.storage."_s, toAPIString(dataType), ".get()"_s);

    Ref storage = storageForType(dataType);
    storage->getValuesForKeys(keys, [callingAPIName, completionHandler = WTFMove(completionHandler)](HashMap<String, String> values, const String& errorMessage) mutable {
        if (!errorMessage.isEmpty())
            completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
        else {
            Ref jsonObject = JSON::Object::create();
            for (auto entry : values)
                jsonObject->setString(entry.key, entry.value);

            completionHandler(jsonObject->toJSONString());
        }
    });
}

void WebExtensionContext::storageGetKeys(WebPageProxyIdentifier webPageProxyIdentifier, WebExtensionDataType dataType, CompletionHandler<void(Expected<Vector<String>, WebExtensionError>&&)>&& completionHandler)
{
    auto callingAPIName = makeString("browser.storage."_s, toAPIString(dataType), ".getKeys()"_s);

    Ref storage = storageForType(dataType);
    storage->getAllKeys([callingAPIName, completionHandler = WTFMove(completionHandler)](Vector<String> keys, const String& errorMessage) mutable {
        if (!errorMessage.isEmpty())
            completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
        else
            completionHandler(keys);
    });
}

void WebExtensionContext::storageGetBytesInUse(WebPageProxyIdentifier webPageProxyIdentifier, WebExtensionDataType dataType, const Vector<String>& keys, CompletionHandler<void(Expected<size_t, WebExtensionError>&&)>&& completionHandler)
{
    auto callingAPIName = makeString("browser.storage."_s, toAPIString(dataType), ".getBytesInUse()"_s);

    Ref storage = storageForType(dataType);
    storage->getStorageSizeForKeys(keys, [callingAPIName, completionHandler = WTFMove(completionHandler)](size_t size, const String& errorMessage) mutable {
        if (!errorMessage.isEmpty())
            completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
        else
            completionHandler(size);
    });
}

void WebExtensionContext::storageSet(WebPageProxyIdentifier webPageProxyIdentifier, WebExtensionDataType dataType, const String& dataJSON, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    auto callingAPIName = makeString("browser.storage."_s, toAPIString(dataType), ".set()"_s);

    HashMap<String, String> data = { };
    if (RefPtr json = JSON::Value::parseJSON(dataJSON); json->asObject()) {
        for (const auto& key : json->asObject()->keys())
            data.set(key, json->asObject()->getString(key));
    }

    Ref storage = storageForType(dataType);
    storage->getStorageSizeForAllKeys(data, [this, protectedThis = Ref { *this }, callingAPIName, dataType, data = WTFMove(data), completionHandler = WTFMove(completionHandler)](size_t size, int numberOfKeys, HashMap<String, String> existingKeysAndValues, const String& errorMessage) mutable {
        if (!errorMessage.isEmpty()) {
            completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
            return;
        }

        if (size > quotaForStorageType(dataType)) {
            completionHandler(toWebExtensionError(callingAPIName, nullString(), "exceeded storage quota"_s));
            return;
        }

        if (dataType == WebExtensionDataType::Sync && static_cast<size_t>(numberOfKeys) > webExtensionStorageAreaSyncMaximumItems) {
            completionHandler(toWebExtensionError(callingAPIName, nullString(), "exceeded maximum number of items"_s));
            return;
        }

        Ref storage = storageForType(dataType);
        storage->setKeyedData(data, [this, protectedThis = Ref { *this }, callingAPIName, data, dataType, existingKeysAndValues = WTFMove(existingKeysAndValues), completionHandler = WTFMove(completionHandler)](Vector<String> keysSuccessfullySet, const String& errorMessage) mutable {
            if (!errorMessage.isEmpty())
                completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
            else
                completionHandler({ });

            // Only fire an onChanged event for the keys that were successfully set.
            if (!keysSuccessfullySet.size())
                return;

            if (keysSuccessfullySet.size() != data.size()) {
                for (const auto& key : data.keys()) {
                    if (!keysSuccessfullySet.contains(key))
                        data.remove(key);
                }
            }

            fireStorageChangedEventIfNeeded(existingKeysAndValues, data, dataType);
        });
    });
}

void WebExtensionContext::storageRemove(WebPageProxyIdentifier webPageProxyIdentifier, WebExtensionDataType dataType, const Vector<String>& keys, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    auto callingAPIName = makeString("browser.storage."_s, toAPIString(dataType), ".remove()"_s);

    Ref storage = storageForType(dataType);
    storage->getValuesForKeys(keys, [this, protectedThis = Ref { *this }, callingAPIName, keys, dataType, completionHandler = WTFMove(completionHandler)](HashMap<String, String> oldValuesAndKeys, const String& errorMessage) mutable {
        if (!errorMessage.isEmpty()) {
            completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
            return;
        }

        Ref storage = storageForType(dataType);
        storage->deleteValuesForKeys(keys, [this, protectedThis = Ref { *this }, callingAPIName, dataType, oldValuesAndKeys = WTFMove(oldValuesAndKeys), completionHandler = WTFMove(completionHandler)](const String& errorMessage) mutable {
            if (!errorMessage.isEmpty()) {
                completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
                return;
            }

            fireStorageChangedEventIfNeeded(oldValuesAndKeys, { }, dataType);

            completionHandler({ });
        });
    });
}

void WebExtensionContext::storageClear(WebPageProxyIdentifier webPageProxyIdentifier, WebExtensionDataType dataType, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    auto callingAPIName = makeString("browser.storage."_s, toAPIString(dataType), ".clear()"_s);

    Ref storage = storageForType(dataType);
    storage->getValuesForKeys({ }, [this, protectedThis = Ref { *this }, callingAPIName, dataType, completionHandler = WTFMove(completionHandler)](HashMap<String, String> oldValuesAndKeys, const String& errorMessage) mutable {
        if (!errorMessage.isEmpty()) {
            completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
            return;
        }

        Ref storage = storageForType(dataType);
        storage->deleteDatabase([this, protectedThis = Ref { *this }, callingAPIName, oldValuesAndKeys = WTFMove(oldValuesAndKeys), dataType, completionHandler = WTFMove(completionHandler)](const String& errorMessage) mutable {
            if (!errorMessage.isEmpty()) {
                completionHandler(toWebExtensionError(callingAPIName, nullString(), errorMessage));
                return;
            }

            fireStorageChangedEventIfNeeded(oldValuesAndKeys, { }, dataType);

            completionHandler({ });
        });
    });
}

void WebExtensionContext::storageSetAccessLevel(WebPageProxyIdentifier webPageProxyIdentifier, WebExtensionDataType dataType, const WebExtensionStorageAccessLevel accessLevel, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    setSessionStorageAllowedInContentScripts(accessLevel == WebExtensionStorageAccessLevel::TrustedAndUntrustedContexts);

    completionHandler({ });
}

void WebExtensionContext::fireStorageChangedEventIfNeeded(HashMap<String, String> oldKeysAndValues, HashMap<String, String> newKeysAndValues, WebExtensionDataType dataType)
{
    static constexpr auto newValueKey = "newValue"_s;
    static constexpr auto oldValueKey = "oldValue"_s;

    if (!oldKeysAndValues.size() && !newKeysAndValues.size())
        return;

    RefPtr changedData = JSON::Object::create();

    // Process new or changed keys
    for (auto entry : newKeysAndValues) {
        const auto& key = entry.key;
        const auto& value = entry.value;

        String oldValue = oldKeysAndValues.get(key);

        if (oldValue.isEmpty() || oldValue != value) {
            RefPtr parsedNewValue = JSON::Value::parseJSON(value);
            RefPtr parsedOldValue = JSON::Value::parseJSON(oldValue);
            RefPtr data = JSON::Object::create();
            if (parsedOldValue)
                data->setValue(oldValueKey, *parsedOldValue);
            data->setValue(newValueKey, *parsedNewValue);
            changedData->setObject(key, *data);
        }
    }

    // Process removed keys.
    for (auto entry : oldKeysAndValues) {
        const auto& key = entry.key;
        const auto& value = entry.value;

        if (!newKeysAndValues.contains(key)) {
            RefPtr data = JSON::Object::create();
            data->setValue(oldValueKey, *(JSON::Value::parseJSON(value)));
            changedData->setObject(key, *data);
        }
    }

    if (!changedData->size())
        return;

    constexpr auto type = WebExtensionEventListenerType::StorageOnChanged;
    auto jsonString = changedData->toJSONString();

    // Unlike other extension events which are only dispatched to the web process that hosts all the extension-related web views (background page, popup, full page extension content),
    // content scripts are allowed to listen to storage.onChanged events.
    sendToContentScriptProcessesForEvent(type, Messages::WebExtensionContextProxy::DispatchStorageChangedEvent(jsonString, dataType, WebExtensionContentWorldType::ContentScript));

    wakeUpBackgroundContentIfNecessaryToFireEvents({ type }, [=, this, protectedThis = Ref { *this }] {
        sendToProcessesForEvent(type, Messages::WebExtensionContextProxy::DispatchStorageChangedEvent(jsonString, dataType, WebExtensionContentWorldType::Main));
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)

