/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#include "CDMLogging.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "CDMKeySystemConfiguration.h"
#include "CDMMediaCapability.h"
#include "CDMMessageType.h"
#include "CDMRestrictions.h"
#include "JSMediaKeyEncryptionScheme.h"
#include "JSMediaKeyMessageEvent.h"
#include "JSMediaKeyMessageType.h"
#include "JSMediaKeySessionType.h"
#include "JSMediaKeyStatusMap.h"
#include "JSMediaKeysRequirement.h"
#include <wtf/JSONValues.h>

namespace WebCore {

static Ref<JSON::Object> toJSONObject(const CDMMediaCapability& capability)
{
    auto object = JSON::Object::create();
    object->setString("contentType"_s, capability.contentType);
    object->setString("robustness"_s, capability.robustness);
    if (capability.encryptionScheme)
        object->setString("encryptionScheme"_s, convertEnumerationToString(capability.encryptionScheme.value()));
    return object;
}

static Ref<JSON::Object> toJSONObject(const CDMRestrictions& restrictions)
{
    auto object = JSON::Object::create();
    object->setBoolean("distinctiveIdentifierDenied"_s, restrictions.distinctiveIdentifierDenied);
    object->setBoolean("persistentStateDenied"_s, restrictions.persistentStateDenied);
    auto deniedSessionTypes = JSON::Array::create();
    for (auto type : restrictions.deniedSessionTypes)
        deniedSessionTypes->pushString(convertEnumerationToString(type));
    object->setArray("deniedSessionTypes"_s, WTFMove(deniedSessionTypes));
    return object;
}

static Ref<JSON::Object> toJSONObject(const CDMKeySystemConfiguration& configuration)
{
    auto object = JSON::Object::create();
    object->setString("label"_s, configuration.label);

    auto initDataTypes = JSON::Array::create();
    for (auto initDataType : configuration.initDataTypes)
        initDataTypes->pushString(initDataType);
    object->setArray("initDataTypes"_s, WTFMove(initDataTypes));

    auto audioCapabilities = JSON::Array::create();
    for (auto capability : configuration.audioCapabilities)
        audioCapabilities->pushObject(toJSONObject(capability));
    object->setArray("audioCapabilities"_s, WTFMove(audioCapabilities));

    auto videoCapabilities = JSON::Array::create();
    for (auto capability : configuration.videoCapabilities)
        videoCapabilities->pushObject(toJSONObject(capability));
    object->setArray("videoCapabilities"_s, WTFMove(videoCapabilities));

    object->setString("distinctiveIdentifier"_s, convertEnumerationToString(configuration.distinctiveIdentifier));
    object->setString("persistentState"_s, convertEnumerationToString(configuration.persistentState));

    auto sessionTypes = JSON::Array::create();
    for (auto type : configuration.sessionTypes)
        sessionTypes->pushString(convertEnumerationToString(type));
    object->setArray("sessionTypes"_s, WTFMove(sessionTypes));

    return object;
}

static String toJSONString(const CDMKeySystemConfiguration& configuration)
{
    return toJSONObject(configuration)->toJSONString();
}

static String toJSONString(const CDMMediaCapability& capability)
{
    return toJSONObject(capability)->toJSONString();
}

static String toJSONString(const CDMRestrictions& restrictions)
{
    return toJSONObject(restrictions)->toJSONString();
}

}

namespace WTF {

String LogArgument<WebCore::CDMKeySystemConfiguration>::toString(const WebCore::CDMKeySystemConfiguration& configuration)
{
    return toJSONString(configuration);
}

String LogArgument<WebCore::CDMMediaCapability>::toString(const WebCore::CDMMediaCapability& capability)
{
    return toJSONString(capability);
}

String LogArgument<WebCore::CDMRestrictions>::toString(const WebCore::CDMRestrictions& restrictions)
{
    return toJSONString(restrictions);
}

String LogArgument<WebCore::CDMEncryptionScheme>::toString(const WebCore::CDMEncryptionScheme& type)
{
    return convertEnumerationToString(type);
}

String LogArgument<WebCore::CDMKeyStatus>::toString(const WebCore::CDMKeyStatus& type)
{
    return convertEnumerationToString(type);
}

String LogArgument<WebCore::CDMMessageType>::toString(const WebCore::CDMMessageType& type)
{
    return convertEnumerationToString(type);
}

String LogArgument<WebCore::CDMRequirement>::toString(const WebCore::CDMRequirement& type)
{
    return convertEnumerationToString(type);
}

String LogArgument<WebCore::CDMSessionType>::toString(const WebCore::CDMSessionType& type)
{
    return convertEnumerationToString(type);
}

}

#endif // ENABLE(ENCRYPTED_MEDIA)
