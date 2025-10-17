/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#include "UserData.h"

#include "APIArray.h"
#include "APIData.h"
#include "APIDictionary.h"
#include "APIError.h"
#include "APIFrameHandle.h"
#include "APIGeometry.h"
#include "APINumber.h"
#include "APIPageHandle.h"
#include "APISerializedScriptValue.h"
#include "APIString.h"
#include "APIURL.h"
#include "APIURLRequest.h"
#include "APIURLResponse.h"
#include "APIUserContentURLPattern.h"
#include "ArgumentCoders.h"
#include "Encoder.h"
#include "WebImage.h"
#include <WebCore/ShareableBitmap.h>
#include <wtf/CheckedArithmetic.h>

namespace WebKit {

UserData::UserData() = default;

UserData::UserData(RefPtr<API::Object>&& object)
    : m_object(WTFMove(object))
{
}

UserData::~UserData() = default;

static bool shouldTransform(const API::Object& object, const UserData::Transformer& transformer)
{
    if (object.type() == API::Object::Type::Array) {
        const auto& array = static_cast<const API::Array&>(object);

        for (const auto& element : array.elements()) {
            if (!element)
                continue;

            if (shouldTransform(*element, transformer))
                return true;
        }
    }

    if (object.type() == API::Object::Type::Dictionary) {
        const auto& dictionary = static_cast<const API::Dictionary&>(object);

        for (const auto& keyValuePair : dictionary.map()) {
            if (!keyValuePair.value)
                continue;

            if (shouldTransform(Ref { *keyValuePair.value }, transformer))
                return true;
        }
    }

    return transformer.shouldTransformObject(object);
}

static RefPtr<API::Object> transformGraph(API::Object& object, const UserData::Transformer& transformer)
{
    if (object.type() == API::Object::Type::Array) {
        auto& array = downcast<API::Array>(object);

        auto elements = array.elements().map([&](auto& element) -> RefPtr<API::Object> {
            if (!element)
                return nullptr;
            return transformGraph(*element, transformer);
        });
        return API::Array::create(WTFMove(elements));
    }

    if (object.type() == API::Object::Type::Dictionary) {
        auto& dictionary = downcast<API::Dictionary>(object);

        API::Dictionary::MapType map;
        for (const auto& keyValuePair : dictionary.map()) {
            RefPtr value = keyValuePair.value;
            if (!value)
                map.add(keyValuePair.key, nullptr);
            else
                map.add(keyValuePair.key, transformGraph(*value, transformer));
        }
        return API::Dictionary::create(WTFMove(map));
    }

    return transformer.transformObject(object);
}

RefPtr<API::Object> UserData::transform(API::Object* object, const Transformer& transformer)
{
    if (!object)
        return nullptr;

    if (!shouldTransform(*object, transformer))
        return object;

    return transformGraph(*object, transformer);
}

} // namespace WebKit
